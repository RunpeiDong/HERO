import SimulationWorker from './worker.js?worker&inline';
import { TabletopRenderer } from './rendering.js';

function cancellation(message, code) {
  const error = new Error(message);
  error.name = 'AbortError'; error.code = code;
  return error;
}

/** A local worker owns physics and inference; camera gestures stay on this thread. */
export class ClientBridge {
  /** Worker /init options. `?carton_grasp=auto|end_face|crotch|spine|spine90` on the page URL pins the carton grasp
   * variant; the worker validates the value and keeps it across Reset through the saved snapshot. */
  static initOptions(href = document.baseURI) {
    let cartonGrasp = null;
    try { cartonGrasp = new URL(href).searchParams.get('carton_grasp'); } catch { /* no query string to read */ }
    return {
      assetsBase: new URL('./', href).href,
      embedded: Boolean(globalThis.__TABLETOP_ASSETS__),
      ...(cartonGrasp ? {carton_grasp: cartonGrasp} : {}),
    };
  }
  constructor({mainCanvas, egoCanvas, onState, onProgress, onError, onIKActivity}) {
    Object.assign(this, {onState, onProgress, onError, onIKActivity});
    this.renderer = new TabletopRenderer({mainCanvas, egoCanvas});
    this.pending = new Map();
    this.sequence = 0;
    this.disposed = false;
    this.lastResetSnapshot = null;
    this.createWorker();
    this.render = timestamp => {
      if (this.disposed) return;
      this.renderer.render?.(timestamp);
      this.animationFrame = requestAnimationFrame(this.render);
    };
    this.animationFrame = requestAnimationFrame(this.render);
  }

  createWorker() {
    const worker = new SimulationWorker({name: 'Tabletop simulation'});
    this.worker = worker; this.workerFailure = null;
    worker.onmessage = event => {
      if (this.disposed || this.worker !== worker) return;
      this.receive(event.data, worker).catch(error => this.failWorker(error, worker));
    };
    worker.onerror = event => this.failWorker(new Error(event.message || 'The local simulation stopped unexpectedly.'), worker);
  }

  failWorker(error, worker) {
    if (this.disposed || this.worker !== worker) return;
    this.workerFailure = error;
    this.renderer.resetPresentation();
    for (const {reject} of this.pending.values()) reject(error);
    this.pending.clear();
    this.onIKActivity?.(false);
    this.onError?.(error.message);
  }

  retireWorker(error) {
    const worker = this.worker;
    this.worker = null;
    this.renderer.resetPresentation();
    if (worker) {
      worker.onmessage = null; worker.onerror = null;
      worker.terminate();
    }
    for (const {reject} of this.pending.values()) reject(error);
    this.pending.clear();
    this.onIKActivity?.(false);
  }

  async receive(message, worker = this.worker) {
    if (this.disposed || !worker || this.worker !== worker) return;
    if (message.type === 'asset') {
      try {
        const encoded = globalThis.__TABLETOP_ASSETS__?.[message.path];
        if (!encoded) throw new Error(`The packaged asset is missing: ${message.path}`);
        const bytes = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
        worker.postMessage({type: 'asset', id: message.id, bytes}, [bytes.buffer]);
      } catch (error) {
        worker.postMessage({type: 'asset', id: message.id, error: error.message});
      }
      return;
    }
    if (message.type === 'progress') this.onProgress?.(message.message);
    if (message.type === 'ik_activity') this.onIKActivity?.(message.active);
    if (message.type === 'model') {
      const camera = this.preserveCameraOnBuild && this.renderer.cameraOptions
        ? structuredClone(this.renderer.cameraOptions) : null;
      this.renderer.build(message.description);
      if (camera) this.renderer.setCamera(camera);
      this.preserveCameraOnBuild = false;
    }
    if (message.type === 'frame') this.renderer.enqueueSnapshot(message.frame, message.preview);
    if (message.type === 'state') {
      if (message.state?.reset_snapshot) this.lastResetSnapshot = structuredClone(message.state.reset_snapshot);
      this.onState?.(message.state);
    }
    if (message.type === 'error') { this.onIKActivity?.(false); this.onError?.(message.message); }
    if (message.type === 'response') {
      const pending = this.pending.get(message.id);
      if (!pending || pending.worker !== worker) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error));
      else {
        if (message.result?.reset_snapshot) this.lastResetSnapshot = structuredClone(message.result.reset_snapshot);
        pending.resolve(message.result);
      }
    }
  }

  request(path, body) {
    if (path === '/api/reset') return this.reset();
    return this.sendRequest(path, body);
  }

  sendRequest(path, body) {
    if (this.disposed) return Promise.reject(cancellation('The demo has closed.', 'DEMO_CLOSED'));
    if (this.workerFailure) return Promise.reject(this.workerFailure);
    const worker = this.worker;
    if (!worker) return Promise.reject(new Error('The local simulation is not running.'));
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      this.pending.set(id, {resolve, reject, worker});
      try { worker.postMessage({type: 'request', id, path, body}); }
      catch (error) { this.pending.delete(id); reject(error); }
    });
  }

  async start() {
    this.initOptions = ClientBridge.initOptions();
    return this.sendRequest('/init', this.initOptions);
  }

  async reset() {
    if (this.disposed) throw cancellation('The demo has closed.', 'DEMO_CLOSED');
    const restore = this.lastResetSnapshot ? structuredClone(this.lastResetSnapshot) : undefined;
    // A synchronous IK solve cannot service a cancel message. Terminate from
    // this thread instead of putting Reset behind the worker's command queue.
    this.retireWorker(cancellation('The current attempt was reset.', 'SIMULATION_RESET'));
    this.preserveCameraOnBuild = true;
    this.createWorker();
    this.onProgress?.('Resetting the simulation…');
    this.initOptions ??= ClientBridge.initOptions();
    return this.sendRequest('/init', {...this.initOptions, restore});
  }

  setCamera(options) { this.renderer.setCamera(options); }
  async saveEgoImage() {
    const value = await this.renderer.egoImage();
    const url = typeof value === 'string' ? value : URL.createObjectURL(value);
    const link = document.createElement('a');
    link.href = url; link.download = 'tabletop-first-person.png'; link.click();
    if (typeof value !== 'string') setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.retireWorker(cancellation('The demo has closed.', 'DEMO_CLOSED'));
    this.renderer.dispose();
  }
}
