// Shared by the console and SYSTEM ONE: size and purge of the Transformers.js model cache.
// One origin holds both models, so purging from either page clears both.
const NAME = 'transformers-cache', NOTE = 'delos-cache-cleared';
export const fmt = b => b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`;
const h = (tag, className, text) => { const n = document.createElement(tag); if (className) n.className = className; if (text) n.textContent = text; return n; };

export async function cachedBytes() {
  if (typeof caches === 'undefined' || !(await caches.has(NAME))) return 0;
  const cache = await caches.open(NAME), requests = await cache.keys();
  let total = 0, exact = true;
  for (const request of requests) {
    const n = Number((await cache.match(request))?.headers.get('content-length'));
    if (n > 0) total += n; else exact = false;
  }
  // ponytail: if a stored response lacks content-length, report the origin's whole storage usage (an overestimate).
  return !requests.length ? 0 : exact ? total : (await navigator.storage?.estimate?.())?.usage ?? total;
}

// Status text plus the dot: cyan and glowing only while online.
export function setStatus(text, online = false) {
  document.getElementById('system-status').textContent = text;
  document.querySelector('.status-dot').classList.toggle('offline', !online);
}

// One-shot "CACHE CLEARED · 1.1 GB reclaimed." line for the gate after the purge reload.
export function takeNote() {
  const size = sessionStorage.getItem(NOTE);
  sessionStorage.removeItem(NOTE);
  return size ? `CACHE CLEARED · ${size} reclaimed.\n` : '';
}

// Wires #cache-clear and builds the confirmation dialog. release() must free the loaded model first.
export function mountCache({ release = async () => {} } = {}) {
  const button = document.getElementById('cache-clear');
  let busy = false, bytes = 0;
  const paint = () => { button.disabled = busy || !bytes; button.textContent = bytes ? `CLEAR CACHE · ${fmt(bytes)}` : 'CACHE EMPTY'; };
  const refresh = async () => { bytes = await cachedBytes().catch(() => 0); paint(); };

  const dialog = h('dialog', 'cache-dialog'), title = h('h2', '', 'PURGE HOST CACHE'), size = h('p', 'cache-size'), error = h('p', 'cache-error');
  const body = h('p', '', 'Deletes both Delos models (Qwen3.5-0.8B console, OneJev-0.8B SYSTEM ONE) from this browser. The cache is shared, so both pages go offline; LOAD HOST downloads them again.');
  const cancel = h('button', '', 'CANCEL'), purge = h('button', 'danger', 'PURGE');
  cancel.type = purge.type = 'button'; cancel.autofocus = true;
  title.id = 'cache-title'; dialog.setAttribute('aria-labelledby', title.id);
  const actions = h('div', 'cache-actions'); actions.append(cancel, purge);
  dialog.append(title, size, body, error, actions);
  document.body.append(dialog);

  cancel.addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => button.focus());
  purge.addEventListener('click', async () => {
    purge.disabled = cancel.disabled = true; error.textContent = '';
    try {
      await release();
      await caches.delete(NAME);
      sessionStorage.setItem(NOTE, fmt(bytes));
      location.reload();
    } catch (e) { error.textContent = `Purge failed: ${e.message}`; purge.disabled = cancel.disabled = false; }
  });
  button.addEventListener('click', async () => {
    await refresh();
    if (!bytes) return;
    size.textContent = `${fmt(bytes)} · ${bytes.toLocaleString('en-US')} bytes on this device`; error.textContent = '';
    dialog.showModal();
  });
  refresh();
  return { refresh, busy: flag => { busy = flag; paint(); } };
}
