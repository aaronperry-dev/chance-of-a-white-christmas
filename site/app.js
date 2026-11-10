/* White Christmas dashboard.
 * The coloured surface is NOAA's own cached map service (ArcGIS tiles) drawn in its native
 * tile grid and projection, ESRI:102009 (North America Lambert Conformal Conic).
 * Everything else (stations, legend, optional state lines / labels) comes from ./data/*. */
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
  const optional = (u) => getJSON(u).catch(() => null);
  const pct = (x, d = 0) => (100 * x).toFixed(d) + '%';
  const css = (v) => getComputedStyle(root).getPropertyValue(v).trim();
  let theme = root.dataset.theme;

  /* ---------- layout: wide map (cards under the map) is remembered ---------- */
  const layoutEl = $('#layout');
  let wide = false;
  try { wide = localStorage.getItem('wc-wide') === '1'; } catch (e) { /* ignore */ }
  layoutEl.classList.toggle('wide', wide);
  $('#t-wide').checked = wide;

  /* ---------- load ---------- */
  let meta, stations;
  try {
    [meta, stations] = await Promise.all([getJSON('data/meta.json'), getJSON('data/stations.json')]);
  } catch (err) {
    const box = $('#map-error');
    box.hidden = false;
    box.textContent = 'Map data not found. Run "python -m pipeline.build" (or let the GitHub Action run) to generate site/data/. ' + err.message;
    return;
  }
  const [statesGJ, places] = await Promise.all([optional('data/states.geojson'), optional('data/places.json')]);

  /* ---------- static text ---------- */
  if (meta.synthetic) $('#demo-banner').hidden = false;
  $('#period').textContent = meta.period;
  $('#built').textContent = 'Built ' + meta.built_utc + '.';
  $('#credit').textContent = 'Map: ' + meta.tiles.attribution + '.';
  $('#kpi-ge50').textContent = pct(meta.stats.share_ge_50);
  $('#kpi-le10').textContent = pct(meta.stats.share_le_10);
  $('#kpi-n').textContent = meta.n_stations.toLocaleString('en-US');
  if (meta.source && meta.source.url) $('#src-link').href = meta.source.url;
  if (meta.tiles.service) $('#tiles-link').href = meta.tiles.service;

  /* ---------- class helpers ---------- */
  const classes = meta.classes;
  const classIndex = (p) => { let i = 0; while (i < classes.length - 1 && p > classes[i].hi) i++; return i; };  // upper edge inclusive
  const colorOf = (p) => classes[classIndex(p)].color;
  const ft = (m) => Math.round(m * 3.28084);

  /* ---------- CRS: NOAA's tile grid ----------
   * Leaflet zoom 0 is an extra, coarser level (2x the service's level 0) so the whole of the
   * contiguous U.S. fits in a narrow window; its tiles are the service's level 0, scaled down.
   * Leaflet zoom z = 1..5 is the service's level z-1. A last, finer step (zoom 6) is overzoom: the
   * finest level scaled up 2x, so the view can go a little past what NOAA publishes. */
  const T = meta.tiles;
  const resolutions = [T.resolutions[0] * 2, ...T.resolutions, T.resolutions[T.resolutions.length - 1] / 2];
  const crs = new L.Proj.CRS(meta.crs.name, meta.crs.proj4, { resolutions, origin: T.origin });
  const proj = (ll) => crs.projection.project(ll);
  const unproj = (x, y) => crs.projection.unproject(L.point(x, y));
  const stationXY = stations.map((s) => proj(L.latLng(s.la, s.lo)));     // LCC metres, computed once
  const stationCls = stations.map((s) => classIndex(s.p));

  const map = L.map('map', {
    crs, zoomSnap: 0.25, zoomDelta: 0.5, wheelPxPerZoomLevel: 90,
    minZoom: 0, maxZoom: resolutions.length - 1, attributionControl: false,
  });
  L.control.attribution({ prefix: false }).addAttribution('NOAA NCEI 1991–2020 Normals · Climate.gov').addTo(map);

  let fitZoom = 0;
  function fitView() {
    const [x0, y0, x1, y1] = meta.crs.extent, s = map.getSize();
    const resNeeded = Math.max((x1 - x0) / (s.x - 20), (y1 - y0) / (s.y - 20));
    const z = fitZoom = Math.max(0, Math.floor(crs.zoom(1 / resNeeded) * 4) / 4);
    map.setView(unproj((x0 + x1) / 2, (y0 + y1) / 2), z, { animate: false });
  }
  fitView();

  /* keep panning inside NOAA's constraint extent (LCC metres). Only correct overshoots of more than
   * one screen pixel: panTo() with a sub-pixel offset fires 'moveend' again straight away, which
   * would re-enter this handler forever. */
  map.on('moveend', () => {
    const c = proj(map.getCenter()), [x0, y0, x1, y1] = meta.crs.extent;
    const cx = Math.min(Math.max(c.x, x0), x1), cy = Math.min(Math.max(c.y, y0), y1);
    const tol = 1 / crs.scale(map.getZoom());                       // metres per pixel at this zoom
    if (Math.abs(cx - c.x) > tol || Math.abs(cy - c.y) > tol) map.panTo(unproj(cx, cy));
  });

  /* ---------- panes ---------- */
  const pane = (name, z) => { const p = map.createPane(name); p.style.zIndex = z; return p; };
  pane('lines', 400); pane('stations', 500);
  pane('labels', 650).style.pointerEvents = 'none';

  /* ---------- NOAA tiles ---------- */
  const [fx0, fy0, fx1, fy1] = T.full_extent, ZOFF = -1;
  const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
  const NoaaTiles = L.TileLayer.extend({
    // only request tiles that exist in the service's cache (its full extent), so no 404 storm
    _isValidTile(c) {
      const res = T.resolutions[Math.max(0, c.z + ZOFF)], span = res * T.size;
      return c.x >= Math.floor((fx0 - T.origin[0]) / span) && c.x <= Math.floor((fx1 - T.origin[0]) / span) &&
             c.y >= Math.floor((T.origin[1] - fy1) / span) && c.y <= Math.floor((T.origin[1] - fy0) / span);
    },
  });
  const noaa = new NoaaTiles(T.url, {
    tileSize: T.size, zoomOffset: ZOFF, minNativeZoom: 1, maxNativeZoom: T.resolutions.length,
    minZoom: 0, maxZoom: resolutions.length - 1, noWrap: true, errorTileUrl: BLANK, keepBuffer: 2, className: 'noaa-tiles',
  }).addTo(map);

  /* ---------- optional vector overlays ---------- */
  const lineLayer = statesGJ ? L.geoJSON(statesGJ, { renderer: L.svg({ pane: 'lines' }), interactive: false }) : null;
  const labelLayer = L.layerGroup();
  const labelMarkers = [];
  if (places) {
    const cityMinZoom = [0.5, 0.5, 1.5, 2.5, 3.5];
    const mkLabel = (cls, text, lat, lon, minZoom) => {
      const m = L.marker([lat, lon], { pane: 'labels', interactive: false, keyboard: false,
        icon: L.divIcon({ className: '', iconSize: [0, 0], html: el('span', 'lbl ' + cls, text) }) });
      m._minZoom = minZoom;
      labelMarkers.push(m);
    };
    places.states.forEach((s) => mkLabel('state', s.t, s.la, s.lo, 1));
    places.cities.forEach((c) => { if (c.us || c.r <= 1) mkLabel('city', c.n, c.la, c.lo, cityMinZoom[Math.min(c.r, 4)]); });
  }
  let labelsOn = false;
  const syncLabels = () => labelMarkers.forEach((m) => {
    const on = labelsOn && map.getZoom() >= m._minZoom;
    if (on && !labelLayer.hasLayer(m)) labelLayer.addLayer(m);
    if (!on && labelLayer.hasLayer(m)) labelLayer.removeLayer(m);
  });
  map.on('zoomend', syncLabels);
  if (!lineLayer) $('#t-lines').disabled = true;
  if (!places) $('#t-labels').disabled = true;

  /* ---------- station layer (canvas) ---------- */
  const canvas = L.canvas({ pane: 'stations', padding: 0.5 });
  const stationLayer = L.layerGroup();
  const tip = L.tooltip({ direction: 'top', offset: [0, -6] });
  const stationLabel = (s) => s.n + (s.s ? ', ' + s.s : '') + ' — ' + s.p + '%';
  const stationMarkers = stations.map((s) => {
    const m = L.circleMarker([s.la, s.lo], { renderer: canvas, radius: 4.5, weight: 1, opacity: 0.9, fillOpacity: 1, bubblingMouseEvents: false });
    m.on('mouseover', (e) => { tip.setContent(el('span', null, stationLabel(s))); tip.setLatLng(e.latlng); map.openTooltip(tip); });
    m.on('mouseout', () => map.closeTooltip(tip));
    m.on('click', () => selectStation(s, false));
    stationLayer.addLayer(m);
    return m;
  });
  let stationsOn = false;
  const dotRadius = () => { const z = map.getZoom(); return z < 1.25 ? 1.8 : z < 2.25 ? 2.8 : z < 3.25 ? 3.8 : 5; };
  const sizeDots = () => { const r = dotRadius(); stationMarkers.forEach((m) => m.setRadius(r)); };
  map.on('zoomend', sizeDots); sizeDots();

  /* ---------- legend + class bars ---------- */
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
      const s = el('i'); s.style.background = c.color; s.title = c.label; bar.append(s);
      ticks.append(el('span', null, i < classes.length - 1 ? String(c.hi) : ''));
    });
    legendEl.append(bar, ticks);
  }
  new Legend().addTo(map);

  /* ---------- stations by chance, for the part of the map that is in view ----------
   * The map's projected axes are screen-aligned, so the visible area is exactly the rectangle
   * spanned by the container's two opposite corners in LCC metres. */
  let view = { n: 0, counts: classes.map(() => 0) };
  function updateView() {
    const sz = map.getSize();
    const a = proj(map.containerPointToLatLng([0, 0])), b = proj(map.containerPointToLatLng([sz.x, sz.y]));
    const x0 = Math.min(a.x, b.x), x1 = Math.max(a.x, b.x), y0 = Math.min(a.y, b.y), y1 = Math.max(a.y, b.y);
    const counts = classes.map(() => 0);
    let n = 0;
    for (let i = 0; i < stationXY.length; i++) {
      const p = stationXY[i];
      if (p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1) { counts[stationCls[i]]++; n++; }
    }
    view = { n, counts };
    renderClasses();
  }
  function renderClasses() {
    const ul = $('#classes'), tb = $('#class-table tbody'), { n, counts } = view;
    ul.replaceChildren(); tb.replaceChildren();
    const total = stations.length, share = (i) => (n ? counts[i] / n : 0), maxShare = Math.max(...counts.map((_, i) => share(i)), 1e-9);
    $('#view-count').textContent = n === 0 ? 'No stations in the current map view.'
      : n === total ? 'All ' + total.toLocaleString('en-US') + ' stations are in view.'
      : n.toLocaleString('en-US') + ' of ' + total.toLocaleString('en-US') + ' stations are in view.';
    classes.forEach((c, i) => {
      const li = el('li'), sw = el('span', 'sw'), track = el('span', 'track'), bar = el('i');
      sw.style.background = c.color;
      bar.style.width = (100 * share(i) / maxShare).toFixed(1) + '%';
      bar.style.background = c.color;
      track.append(bar);
      li.append(sw, el('span', 'rng-l', c.label), track, el('span', 'val', n ? pct(share(i), 1) : '–'));
      ul.append(li);
      const tr = el('tr'); tr.append(el('td', null, c.label), el('td', null, counts[i].toLocaleString('en-US')), el('td', null, n ? pct(share(i), 1) : '–')); tb.append(tr);
    });
  }
  map.on('moveend resize', updateView);

  /* ---------- nearest station ---------- */
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
  function renderReadout(s, lines) {
    const box = $('#readout');
    box.replaceChildren();
    const top = el('div', 'read-top'), sw = el('span', 'sw');
    sw.style.background = colorOf(s.p);
    top.append(el('span', 'big', s.p + '%'), sw);
    box.append(top, el('div', 'readout-kind', meta.synthetic ? 'Synthetic demo value (not NOAA data)' : 'Chance of ≥1 in. snow on the ground, Dec 25'));
    for (const parts of lines) {
      const d = el('div', 'read-meta');
      parts.forEach((t, i) => d.append(i % 2 ? el('b', null, t) : document.createTextNode(t)));
      box.append(d);
    }
  }
  const stationName = (s) => s.n + (s.s ? ', ' + s.s : '');
  const stationFacts = (s) => {
    const lines = [];
    if (s.e !== null) lines.push(['Elevation: ', ft(s.e).toLocaleString('en-US') + ' ft (' + Math.round(s.e).toLocaleString('en-US') + ' m)', '']);
    if (s.y) lines.push(['Years of record: ', String(s.y), '']);
    return lines;
  };
  function selectPoint(ll) {
    last = { type: 'point', ll };
    setMarker(ll);
    const n = nearestStation(ll.lat, ll.lng);
    if (!n) return;
    renderReadout(n.s, [['Station: ', stationName(n.s), ' — ' + n.mi.toFixed(0) + ' mi from the point you clicked'], ...stationFacts(n.s)]);
  }
  function selectStation(s, fly) {
    last = { type: 'station', s };
    setMarker(L.latLng(s.la, s.lo));
    if (fly) map.flyTo([s.la, s.lo], Math.max(map.getZoom(), 4), { duration: 0.8 });
    renderReadout(s, [['Station: ', stationName(s), ''], ...stationFacts(s), ['ID: ', s.id, '']]);
  }
  map.on('click', (e) => selectPoint(e.latlng));

  /* ---------- theme (page chrome and overlays; NOAA's tiles keep NOAA's colours) ---------- */
  function paint() {
    root.dataset.theme = theme;
    document.querySelectorAll('[data-theme-set]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.themeSet === theme)));
    if (lineLayer) lineLayer.setStyle({ stroke: true, color: css('--border'), weight: 0.9, fill: false, lineJoin: 'round' });
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
  updateView();

  /* ---------- station dots: one switch, on the map and in the toolbar ---------- */
  const stationsBox = $('#t-stations');
  let stationsBtn = null;
  function setStations(on) {
    stationsOn = on;
    on ? stationLayer.addTo(map) : stationLayer.remove();
    stationsBox.checked = on;
    if (stationsBtn) {
      stationsBtn.setAttribute('aria-pressed', String(on));
      stationsBtn.textContent = 'Stations: ' + (on ? 'On' : 'Off');
    }
  }
  const StationsToggle = L.Control.extend({
    options: { position: 'topright' },
    onAdd() {
      stationsBtn = el('button', 'map-toggle'); stationsBtn.type = 'button';
      L.DomEvent.disableClickPropagation(stationsBtn);
      stationsBtn.addEventListener('click', () => setStations(!stationsOn));
      return stationsBtn;
    },
  });
  new StationsToggle().addTo(map);
  setStations(false);
  stationsBox.addEventListener('change', (e) => setStations(e.target.checked));

  /* ---------- toolbar ---------- */
  $('#t-wide').addEventListener('change', (e) => {
    layoutEl.classList.toggle('wide', e.target.checked);
    try { localStorage.setItem('wc-wide', e.target.checked ? '1' : '0'); } catch (x) { /* ignore */ }
    const showingAll = map.getZoom() <= fitZoom + 0.01;           // looking at the whole country?
    map.invalidateSize({ pan: false });
    if (showingAll) fitView();                                    // then use the new space; else keep the user's view
  });
  const toggle = (id, layer) => $(id).addEventListener('change', (e) => { e.target.checked ? layer.addTo(map) : layer.remove(); });
  if (lineLayer) toggle('#t-lines', lineLayer);
  $('#t-labels').addEventListener('change', (e) => {
    labelsOn = e.target.checked;
    labelsOn ? labelLayer.addTo(map) : labelLayer.remove();
    syncLabels();
  });

  /* station search */
  const input = $('#find'), list = $('#find-results');
  let active = -1;
  const closeList = () => { list.hidden = true; list.replaceChildren(); active = -1; input.removeAttribute('aria-activedescendant'); };
  const choose = (s) => { closeList(); input.value = stationName(s); selectStation(s, true); };
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    closeList();
    if (q.length < 2) return;
    const hits = [];
    for (const s of stations) { if ((s.n + ' ' + s.s).toLowerCase().includes(q)) { hits.push(s); if (hits.length === 8) break; } }
    hits.forEach((s, i) => {
      const li = el('li'); li.id = 'fr-' + i; li.setAttribute('role', 'option');
      li.append(el('span', null, stationName(s)), el('span', 'p', s.p + '%'));
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

  window.__wc = { map, crs, meta, noaa, proj, unproj, stationLayer, stations, classIndex, view: () => view };   // handy for debugging / tests
})();
