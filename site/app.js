/* White Christmas dashboard. All data comes from ./data/* written by the Python pipeline.
 * Map CRS = ESRI:102009 (North America Lambert Conformal Conic), same as NOAA's web app. */
(async function () {
  'use strict';

  const root = document.documentElement;
  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;       // never innerHTML for data
    return n;
  };
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + ' -> ' + r.status); return r.json(); });
  const pct = (x, d = 0) => (100 * x).toFixed(d) + '%';
  const css = (v) => getComputedStyle(root).getPropertyValue(v).trim();
  let theme = root.dataset.theme;

  /* ---------- load ---------- */
  let meta, stations, places, statesGJ, contextGJ;
  try {
    [meta, stations, places, statesGJ, contextGJ] = await Promise.all([
      getJSON('data/meta.json'), getJSON('data/stations.json'), getJSON('data/places.json'),
      getJSON('data/states.geojson'), getJSON('data/context.geojson'),
    ]);
  } catch (err) {
    const box = $('#map-error');
    box.hidden = false;
    box.textContent = 'Map data not found. Run "python -m pipeline.build" (or let the GitHub Action run) to generate site/data/. ' + err.message;
    return;
  }

  /* value grid for exact click look-ups (gzip uint8, 255 = no data); optional */
  let values = null;
  try {
    if (!('DecompressionStream' in window)) throw new Error('DecompressionStream unavailable');
    const r = await fetch('data/' + meta.values);
    values = new Uint8Array(await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  } catch (e) { console.warn('click look-up grid unavailable:', e.message); }

  /* ---------- static text ---------- */
  if (meta.synthetic) $('#demo-banner').hidden = false;
  $('#period').textContent = meta.period;
  $('#palette').textContent = meta.palette;
  $('#built').textContent = 'Built ' + meta.built_utc + '.';
  $('#kpi-ge50').textContent = pct(meta.stats.share_ge_50);
  $('#kpi-lt10').textContent = pct(meta.stats.share_lt_10);
  $('#kpi-n').textContent = meta.n_stations.toLocaleString('en-US');
  if (meta.source && meta.source.url) $('#src-link').href = meta.source.url;
  (function methodText() {
    const g = meta.grid, e = meta.estimation;
    let t = 'Station values are estimated on a ' + (g.res / 1000) + ' km grid in the map projection';
    t += ' by inverse-distance weighting of the nearest stations';
    if (e.cv) {
      t += '. Leave-one-out check at the stations: RMSE ' + e.cv.rmse.toFixed(1) + ' percentage points' +
        (meta.synthetic ? ' (synthetic data)' : '');
    }
    $('#method').textContent = t + '.';
  })();

  /* ---------- class helpers ---------- */
  const classes = meta.classes;
  const classIndex = (p) => { let i = 0; while (i < classes.length - 1 && p >= classes[i].hi) i++; return i; };
  const colorOf = (p) => classes[classIndex(p)].colors[theme];
  const ft = (m) => Math.round(m * 3.28084);

  /* ---------- CRS: ESRI:102009 ---------- */
  const crs = new L.Proj.CRS(meta.crs.name, meta.crs.proj4, {
    resolutions: meta.crs.resolutions, origin: [meta.crs.extent[0], meta.crs.extent[3]],
  });
  const g = meta.grid;
  const proj = (ll) => crs.projection.project(ll);
  const unproj = (x, y) => crs.projection.unproject(L.point(x, y));

  const map = L.map('map', {
    crs, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 90,
    minZoom: 1, maxZoom: 7, attributionControl: false,
  });
  L.control.attribution({ prefix: false })
    .addAttribution('NOAA NCEI 1991–2020 Normals · Natural Earth')
    .addTo(map);

  function fitView() {
    const s = map.getSize();
    const resNeeded = Math.max((g.x1 - g.x0) / (s.x - 20), (g.y1 - g.y0) / (s.y - 20));
    const z = Math.floor(crs.zoom(1 / resNeeded) * 4) / 4;
    map.setView(unproj((g.x0 + g.x1) / 2, (g.y0 + g.y1) / 2), z, { animate: false });
  }
  fitView();

  /* keep panning inside NOAA's constraint extent (in LCC metres) */
  map.on('moveend', () => {
    const c = proj(map.getCenter()), [x0, y0, x1, y1] = meta.crs.extent;
    const cx = Math.min(Math.max(c.x, x0), x1), cy = Math.min(Math.max(c.y, y0), y1);
    if (cx !== c.x || cy !== c.y) map.panTo(unproj(cx, cy));
  });

  /* ---------- panes (stacking order) ---------- */
  const pane = (name, z) => { const p = map.createPane(name); p.style.zIndex = z; return p; };
  pane('ctx', 200); pane('land', 210); pane('surface', 300);
  pane('lines', 400); pane('stations', 500);
  pane('labels', 650).style.pointerEvents = 'none';

  /* ---------- image layer placed by its LCC corners (conic-safe) ----------
   * Leaflet's ImageOverlay assumes an axis-aligned lat/lon rectangle, which is
   * wrong in a conic projection; this positions the image by projected corners. */
  const LccImage = L.ImageOverlay.extend({
    initialize(url, ext, options) {
      this._ext = ext;
      L.ImageOverlay.prototype.initialize.call(this, url, [[0, 0], [1, 1]], options);
    },
    _tl() { return unproj(this._ext.x0, this._ext.y1); },
    _br() { return unproj(this._ext.x1, this._ext.y0); },
    _reset() {
      const m = this._map, tl = m.latLngToLayerPoint(this._tl()), br = m.latLngToLayerPoint(this._br());
      L.DomUtil.setPosition(this._image, tl);
      this._image.style.width = (br.x - tl.x) + 'px';
      this._image.style.height = (br.y - tl.y) + 'px';
    },
    _animateZoom(e) {
      const m = this._map;
      L.DomUtil.setTransform(this._image, m._latLngToNewLayerPoint(this._tl(), e.zoom, e.center), m.getZoomScale(e.zoom));
    },
  });

  const surface = new LccImage('data/' + meta.overlay[theme], g, {
    pane: 'surface', opacity: 0.8, interactive: false, alt: 'Probability of a white Christmas',
  }).addTo(map);

  /* ---------- vector context ---------- */
  const ctxLayer = L.geoJSON(contextGJ, { renderer: L.svg({ pane: 'ctx' }), interactive: false }).addTo(map);
  const landLayer = L.geoJSON(statesGJ, { renderer: L.svg({ pane: 'land' }), interactive: false }).addTo(map);
  const lineLayer = L.geoJSON(statesGJ, { renderer: L.svg({ pane: 'lines' }), interactive: false }).addTo(map);

  /* ---------- labels ---------- */
  const labelLayer = L.layerGroup().addTo(map);
  const cityMinZoom = [1, 1, 2.5, 3.5, 4.5];
  const labelMarkers = [];
  const mkLabel = (cls, text, lat, lon, minZoom) => {
    const m = L.marker([lat, lon], { pane: 'labels', interactive: false, keyboard: false,
      icon: L.divIcon({ className: '', iconSize: [0, 0], html: el('span', 'lbl ' + cls, text) }) });
    m._minZoom = minZoom;
    labelMarkers.push(m);
  };
  places.states.forEach((s) => mkLabel('state', s.t, s.la, s.lo, 3));
  places.cities.forEach((c) => { if (c.us || c.r <= 1) mkLabel('city', c.n, c.la, c.lo, cityMinZoom[Math.min(c.r, 4)]); });
  const syncLabels = () => labelMarkers.forEach((m) => {
    const on = map.getZoom() >= m._minZoom;
    if (on && !labelLayer.hasLayer(m)) labelLayer.addLayer(m);
    if (!on && labelLayer.hasLayer(m)) labelLayer.removeLayer(m);
  });
  map.on('zoomend', syncLabels); syncLabels();

  /* ---------- station layer (canvas: thousands of points stay fast) ---------- */
  const canvas = L.canvas({ pane: 'stations', padding: 0.5 });
  const stationLayer = L.layerGroup();
  const tip = L.tooltip({ direction: 'top', offset: [0, -6] });
  const stationLabel = (s) => s.n + (s.s ? ', ' + s.s : '') + ' — ' + s.p + '%';
  const stationMarkers = stations.map((s) => {
    const m = L.circleMarker([s.la, s.lo], {
      renderer: canvas, radius: 4.5, weight: 1, opacity: 0.9, fillOpacity: 1, bubblingMouseEvents: false,
    });
    m.on('mouseover', (e) => { tip.setContent(el('span', null, stationLabel(s))); tip.setLatLng(e.latlng); map.openTooltip(tip); });
    m.on('mouseout', () => map.closeTooltip(tip));
    m.on('click', () => selectStation(s, false));
    stationLayer.addLayer(m);
    return m;
  });

  /* dots shrink when zoomed out so they don't bury the surface */
  const dotRadius = () => { const z = map.getZoom(); return z < 2.5 ? 1.8 : z < 3.5 ? 2.8 : z < 4.5 ? 3.8 : 5; };
  const sizeDots = () => { const r = dotRadius(); stationMarkers.forEach((m) => m.setRadius(r)); };
  map.on('zoomend', sizeDots); sizeDots();

  /* ---------- legend + class bars (re-rendered on theme change) ---------- */
  let legendEl = null;
  const Legend = L.Control.extend({
    options: { position: 'bottomleft' },
    onAdd() { legendEl = el('div', 'legend'); L.DomEvent.disableClickPropagation(legendEl); renderLegend(); return legendEl; },
  });
  function renderLegend() {
    if (!legendEl) return;
    legendEl.replaceChildren(el('div', 't', 'Chance of ≥1 in. snow on the ground, Dec 25'));
    const bar = el('div', 'bar'), ticks = el('div', 'ticks');
    classes.forEach((c, i) => {
      const s = el('i'); s.style.background = c.colors[theme]; s.title = c.label; bar.append(s);
      ticks.append(el('span', null, i < classes.length - 1 ? String(c.hi) : ''));
    });
    legendEl.append(bar, ticks);
  }
  new Legend().addTo(map);

  const shares = meta.stats.share_by_class, maxShare = Math.max(...shares);
  function renderClasses() {
    const ul = $('#classes'), tb = $('#class-table tbody');
    ul.replaceChildren(); tb.replaceChildren();
    classes.forEach((c, i) => {
      const li = el('li'), sw = el('span', 'sw'), track = el('span', 'track'), bar = el('i');
      sw.style.background = c.colors[theme];
      bar.style.width = (100 * shares[i] / maxShare).toFixed(1) + '%';
      bar.style.background = c.colors[theme];
      track.append(bar);
      li.append(sw, el('span', 'rng-l', c.label), track, el('span', 'val', pct(shares[i], 1)));
      ul.append(li);
      const tr = el('tr'); tr.append(el('td', null, c.label), el('td', null, pct(shares[i], 1))); tb.append(tr);
    });
  }

  /* ---------- value look-up (exact, full grid resolution) ---------- */
  function valueAt(latlng) {
    if (!values) return undefined;
    const p = proj(latlng);
    const x = Math.floor((p.x - g.x0) / g.res), y = Math.floor((g.y1 - p.y) / g.res);
    if (x < 0 || y < 0 || x >= g.nx || y >= g.ny) return null;
    const v = values[y * g.nx + x];
    return v === 255 ? null : v;
  }
  function nearestStation(lat, lon) {
    const k = Math.cos((lat * Math.PI) / 180);
    let best = null, bd = Infinity;
    for (const s of stations) {
      const dy = s.la - lat, dx = (s.lo - lon) * k, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = s; }
    }
    if (!best) return null;
    const R = 3958.8, r = Math.PI / 180;
    const a = Math.sin(((best.la - lat) * r) / 2) ** 2 +
      Math.cos(lat * r) * Math.cos(best.la * r) * Math.sin(((best.lo - lon) * r) / 2) ** 2;
    return { s: best, mi: 2 * R * Math.asin(Math.sqrt(a)) };
  }

  /* ---------- readout ---------- */
  let marker = null, last = null;
  function setMarker(ll) {
    if (marker) marker.remove();
    marker = L.circleMarker(ll, { pane: 'stations', radius: 7, weight: 2, color: css('--ring'), fillColor: css('--card'), fillOpacity: 0.9, interactive: false }).addTo(map);
  }
  function renderReadout(p, kind, lines) {
    const box = $('#readout');
    box.replaceChildren();
    if (p === null || p === undefined) {
      box.append(el('p', 'hint', p === null ? 'Outside the mapped contiguous U.S. surface.' : 'Click look-up is unavailable in this browser; stations still work.'));
      for (const parts of lines) {
        const d = el('div', 'read-meta');
        parts.forEach((t, i) => d.append(i % 2 ? el('b', null, t) : document.createTextNode(t)));
        box.append(d);
      }
      return;
    }
    const top = el('div', 'read-top'), sw = el('span', 'sw');
    sw.style.background = colorOf(p);
    top.append(el('span', 'big', p + '%'), sw);
    box.append(top, el('div', 'readout-kind', kind));
    for (const parts of lines) {
      const d = el('div', 'read-meta');
      parts.forEach((t, i) => d.append(i % 2 ? el('b', null, t) : document.createTextNode(t)));
      box.append(d);
    }
  }
  function selectPoint(ll) {
    last = { type: 'point', ll };
    setMarker(ll);
    const n = nearestStation(ll.lat, ll.lng), lines = [];
    if (n) lines.push(['Nearest station: ', n.s.n + (n.s.s ? ', ' + n.s.s : ''), ' — ' + n.mi.toFixed(0) + ' mi away, ' + n.s.p + '%']);
    lines.push(['Location: ', ll.lat.toFixed(2) + '°, ' + ll.lng.toFixed(2) + '°', '']);
    renderReadout(valueAt(ll), 'Estimated surface value', lines);
  }
  function selectStation(s, fly) {
    last = { type: 'station', s };
    setMarker(L.latLng(s.la, s.lo));
    if (fly) map.flyTo([s.la, s.lo], Math.max(map.getZoom(), 4.5), { duration: 0.8 });
    const lines = [['Station: ', s.n + (s.s ? ', ' + s.s : ''), '']];
    if (s.e !== null) lines.push(['Elevation: ', ft(s.e).toLocaleString('en-US') + ' ft (' + Math.round(s.e).toLocaleString('en-US') + ' m)', '']);
    if (s.y) lines.push(['Years of record: ', String(s.y), '']);
    lines.push(['ID: ', s.id, '']);
    renderReadout(s.p, meta.synthetic ? 'Synthetic demo value (not NOAA data)' : 'Station value from NOAA normals', lines);
  }
  map.on('click', (e) => selectPoint(e.latlng));

  /* ---------- theme ---------- */
  function paint() {
    root.dataset.theme = theme;
    document.querySelectorAll('[data-theme-set]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.themeSet === theme)));
    surface.setUrl('data/' + meta.overlay[theme]);
    landLayer.setStyle({ stroke: false, fill: true, fillColor: css('--land'), fillOpacity: 1 });
    ctxLayer.setStyle({ stroke: true, color: css('--ctx-border'), weight: 0.8, fill: true, fillColor: css('--ctx'), fillOpacity: 1 });
    lineLayer.setStyle({ stroke: true, color: css('--border'), weight: 0.8, fill: false, lineJoin: 'round' });
    stationMarkers.forEach((m, i) => m.setStyle({ fillColor: colorOf(stations[i].p), color: css('--ring') }));
    renderLegend(); renderClasses();
    if (last) { last.type === 'point' ? selectPoint(last.ll) : selectStation(last.s, false); }
  }
  document.querySelectorAll('[data-theme-set]').forEach((b) => b.addEventListener('click', () => {
    theme = b.dataset.themeSet;
    try { localStorage.setItem('wc-theme', theme); } catch (e) { /* private mode: ignore */ }
    paint();
  }));
  matchMedia('(prefers-color-scheme: light)').addEventListener?.('change', (e) => {
    let saved = null; try { saved = localStorage.getItem('wc-theme'); } catch (x) {}
    if (!saved) { theme = e.matches ? 'light' : 'dark'; paint(); }
  });
  paint();

  /* ---------- toolbar ---------- */
  $('#t-surface').addEventListener('change', (e) => { e.target.checked ? surface.addTo(map) : surface.remove(); });
  $('#t-stations').addEventListener('change', (e) => { e.target.checked ? stationLayer.addTo(map) : stationLayer.remove(); });
  $('#opacity').addEventListener('input', (e) => surface.setOpacity(e.target.value / 100));

  /* station search */
  const input = $('#find'), list = $('#find-results');
  let active = -1;
  const closeList = () => { list.hidden = true; list.replaceChildren(); active = -1; input.removeAttribute('aria-activedescendant'); };
  const choose = (s) => { closeList(); input.value = s.n + (s.s ? ', ' + s.s : ''); selectStation(s, true); };
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    closeList();
    if (q.length < 2) return;
    const hits = [];
    for (const s of stations) { if ((s.n + ' ' + s.s).toLowerCase().includes(q)) { hits.push(s); if (hits.length === 8) break; } }
    hits.forEach((s, i) => {
      const li = el('li'); li.id = 'fr-' + i; li.setAttribute('role', 'option');
      li.append(el('span', null, s.n + (s.s ? ', ' + s.s : '')), el('span', 'p', s.p + '%'));
      li.addEventListener('mousedown', (ev) => { ev.preventDefault(); choose(s); });
      list.append(li);
    });
    list.hidden = hits.length === 0;
    list._hits = hits;
  });
  input.addEventListener('keydown', (e) => {
    const items = [...list.children];
    if (!items.length) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      active = (active + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.setAttribute('aria-selected', String(i === active)));
      input.setAttribute('aria-activedescendant', items[active].id);
    } else if (e.key === 'Enter' && active >= 0) { e.preventDefault(); choose(list._hits[active]); }
    else if (e.key === 'Escape') { closeList(); }
  });
  input.addEventListener('blur', () => setTimeout(closeList, 120));

  window.__wc = { map, crs, meta, valueAt };   // handy for debugging / tests
})();
