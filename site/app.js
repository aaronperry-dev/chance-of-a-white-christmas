/* White Christmas dashboard. All data comes from ./data/* written by the Python pipeline. */
(async function () {
  'use strict';

  const $ = (s) => document.querySelector(s);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;       // never innerHTML for data
    return n;
  };
  const getJSON = (u) => fetch(u).then((r) => { if (!r.ok) throw new Error(u + ' -> ' + r.status); return r.json(); });
  const pct = (x, d = 0) => (100 * x).toFixed(d) + '%';

  /* ---------- load ---------- */
  let meta, stations, grid;
  try {
    [meta, stations, grid] = await Promise.all([
      getJSON('data/meta.json'), getJSON('data/stations.json'), getJSON('data/grid.json'),
    ]);
  } catch (err) {
    const box = $('#map-error');
    box.hidden = false;
    box.textContent = 'Map data not found. Run "python -m pipeline.build" (or let the GitHub Action run) to generate site/data/. ' + err.message;
    return;
  }

  /* ---------- static text ---------- */
  if (meta.synthetic) $('#demo-banner').hidden = false;
  $('#period').textContent = meta.period;
  $('#method').textContent = 'Station values are interpolated to a ~' + (meta.grid.step * 111 * 0.78).toFixed(1) +
    ' km grid with ' + meta.interpolation + '.';
  $('#palette').textContent = meta.palette;
  $('#built').textContent = 'Built ' + meta.built_utc + '.';
  $('#kpi-ge50').textContent = pct(meta.stats.share_ge_50);
  $('#kpi-lt10').textContent = pct(meta.stats.share_lt_10);
  $('#kpi-n').textContent = meta.n_stations.toLocaleString('en-US');

  /* ---------- class helpers (colours come from the pipeline) ---------- */
  const classes = meta.classes;
  const classIndex = (p) => {
    let i = 0;
    while (i < classes.length - 1 && p >= classes[i].hi) i++;
    return i;
  };
  const colorOf = (p) => classes[classIndex(p)].color;

  /* ---------- side panel: area by class ---------- */
  const shares = meta.stats.share_by_class;
  const maxShare = Math.max(...shares);
  const ul = $('#classes');
  const tbody = $('#class-table tbody');
  classes.forEach((c, i) => {
    const li = el('li');
    const sw = el('span', 'sw'); sw.style.background = c.color;
    const track = el('span', 'track'); const bar = el('i');
    bar.style.width = (100 * shares[i] / maxShare).toFixed(1) + '%';
    bar.style.background = c.color;
    track.append(bar);
    li.append(sw, el('span', 'rng-l', c.label), track, el('span', 'val', pct(shares[i], 1)));
    ul.append(li);

    const tr = el('tr');
    tr.append(el('td', null, c.label), el('td', null, pct(shares[i], 1)));
    tbody.append(tr);
  });

  /* ---------- value grid (click look-up) ---------- */
  const g = meta.grid;
  const bin = atob(grid.b64);
  const vals = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) vals[i] = bin.charCodeAt(i);
  const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
  function valueAt(lat, lon) {
    const x = Math.floor((lon - g.west) / g.step);
    const y = Math.floor((g.y_top - mercY(lat)) / g.dy);
    if (x < 0 || y < 0 || x >= g.nx || y >= g.ny) return null;
    const v = vals[y * g.nx + x];
    return v === 255 ? null : v;
  }

  function nearestStation(lat, lon) {
    const k = Math.cos((lat * Math.PI) / 180);
    let best = null, bd = Infinity;
    for (const s of stations) {
      const dy = s.la - lat, dx = (s.lo - lon) * k;
      const d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = s; }
    }
    if (!best) return null;
    const R = 3958.8;                                     // miles
    const toRad = Math.PI / 180;
    const a = Math.sin(((best.la - lat) * toRad) / 2) ** 2 +
      Math.cos(lat * toRad) * Math.cos(best.la * toRad) * Math.sin(((best.lo - lon) * toRad) / 2) ** 2;
    return { s: best, mi: 2 * R * Math.asin(Math.sqrt(a)) };
  }

  /* ---------- map ---------- */
  const map = L.map('map', {
    zoomSnap: 0.25, minZoom: 3, maxZoom: 11, worldCopyJump: false,
    maxBounds: [[10, -145], [60, -45]], maxBoundsViscosity: 0.7,
    attributionControl: false,
  });
  L.control.attribution({ prefix: false }).addTo(map);
  map.fitBounds(meta.bounds, { padding: [4, 4] });
  map.createPane('surface').style.zIndex = 300;
  map.createPane('stations').style.zIndex = 500;
  map.createPane('labels').style.zIndex = 650;
  map.getPane('labels').style.pointerEvents = 'none';

  const CARTO = 'https://{s}.basemaps.cartocdn.com/';
  const attribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';
  L.tileLayer(CARTO + 'dark_nolabels/{z}/{x}/{y}{r}.png', { subdomains: 'abcd', maxZoom: 20, attribution }).addTo(map);
  L.tileLayer(CARTO + 'dark_only_labels/{z}/{x}/{y}{r}.png', { subdomains: 'abcd', maxZoom: 20, pane: 'labels' }).addTo(map);

  const surface = L.imageOverlay('data/overlay.png', meta.bounds, {
    pane: 'surface', opacity: 0.8, interactive: false, alt: 'Probability of a white Christmas surface',
  }).addTo(map);

  /* on-map legend */
  const Legend = L.Control.extend({
    options: { position: 'bottomleft' },
    onAdd() {
      const box = el('div', 'legend');
      box.append(el('div', 't', 'Chance of ≥1 in. snow on the ground, Dec 25'));
      const bar = el('div', 'bar');
      classes.forEach((c) => { const i = el('i'); i.style.background = c.color; i.title = c.label; bar.append(i); });
      const ticks = el('div', 'ticks');
      classes.forEach((c, i) => ticks.append(el('span', null, i < classes.length - 1 ? String(c.hi) : '')));
      box.append(bar, ticks);
      L.DomEvent.disableClickPropagation(box);
      return box;
    },
  });
  new Legend().addTo(map);

  /* stations layer (canvas: thousands of points stay fast) */
  const canvas = L.canvas({ pane: 'stations', padding: 0.5 });
  const stationLayer = L.layerGroup();
  const tip = L.tooltip({ direction: 'top', offset: [0, -6], pane: 'tooltipPane' });
  const stationLabel = (s) => s.n + (s.s ? ', ' + s.s : '') + ' — ' + s.p + '%';
  for (const s of stations) {
    const m = L.circleMarker([s.la, s.lo], {
      renderer: canvas, radius: 4.5, weight: 1, color: '#ffffff', opacity: 0.85,
      fillColor: colorOf(s.p), fillOpacity: 1, bubblingMouseEvents: false,
    });
    m.on('mouseover', (e) => { tip.setContent(el('span', null, stationLabel(s))); tip.setLatLng(e.latlng); map.openTooltip(tip); });
    m.on('mouseout', () => map.closeTooltip(tip));
    m.on('click', () => selectStation(s, false));
    stationLayer.addLayer(m);
  }

  /* ---------- readout ---------- */
  let marker = null;
  function setMarker(lat, lon) {
    if (marker) marker.remove();
    marker = L.circleMarker([lat, lon], {
      pane: 'stations', radius: 7, weight: 2, color: '#ffffff', fillColor: '#121417', fillOpacity: 0.9, interactive: false,
    }).addTo(map);
  }

  function renderReadout(p, kind, lines) {
    const box = $('#readout');
    box.replaceChildren();
    if (p === null) {
      box.append(el('p', 'hint', 'Outside the mapped contiguous U.S. surface.'));
      return;
    }
    const top = el('div', 'read-top');
    const sw = el('span', 'sw'); sw.style.background = colorOf(p);
    top.append(el('span', 'big', p + '%'), sw);
    box.append(top, el('div', 'readout-kind', kind));
    for (const parts of lines) {
      const d = el('div', 'read-meta');
      parts.forEach((t, i) => d.append(i % 2 ? el('b', null, t) : document.createTextNode(t)));
      box.append(d);
    }
  }

  function selectPoint(lat, lon) {
    setMarker(lat, lon);
    const p = valueAt(lat, lon);
    const n = nearestStation(lat, lon);
    const lines = [];
    if (n) {
      lines.push(['Nearest station: ', n.s.n + (n.s.s ? ', ' + n.s.s : ''), ' — ' + n.mi.toFixed(0) + ' mi away, ' + n.s.p + '%']);
    }
    lines.push(['Location: ', lat.toFixed(2) + '°, ' + lon.toFixed(2) + '°', '']);
    renderReadout(p, 'Interpolated surface value', lines);
  }

  function selectStation(s, fly) {
    setMarker(s.la, s.lo);
    if (fly) map.flyTo([s.la, s.lo], Math.max(map.getZoom(), 7), { duration: 0.8 });
    const lines = [['Station: ', s.n + (s.s ? ', ' + s.s : ''), '']];
    if (s.e !== null) lines.push(['Elevation: ', Math.round(s.e).toLocaleString('en-US') + ' ft', '']);
    lines.push(['ID: ', s.id, '']);
    renderReadout(s.p, meta.synthetic ? 'Synthetic demo value (not NOAA data)' : 'Station value from NOAA normals', lines);
  }

  map.on('click', (e) => selectPoint(e.latlng.lat, e.latlng.lng));

  /* ---------- toolbar ---------- */
  $('#t-surface').addEventListener('change', (e) => { e.target.checked ? surface.addTo(map) : surface.remove(); });
  $('#t-stations').addEventListener('change', (e) => { e.target.checked ? stationLayer.addTo(map) : stationLayer.remove(); });
  $('#opacity').addEventListener('input', (e) => surface.setOpacity(e.target.value / 100));

  /* station search */
  const input = $('#find'), list = $('#find-results');
  let active = -1;
  function closeList() { list.hidden = true; list.replaceChildren(); active = -1; input.removeAttribute('aria-activedescendant'); }
  function choose(s) { closeList(); input.value = s.n + (s.s ? ', ' + s.s : ''); selectStation(s, true); }
  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    closeList();
    if (q.length < 2) return;
    const hits = [];
    for (const s of stations) {
      if ((s.n + ' ' + s.s).toLowerCase().includes(q)) { hits.push(s); if (hits.length === 8) break; }
    }
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
    } else if (e.key === 'Enter' && active >= 0) {
      e.preventDefault(); choose(list._hits[active]);
    } else if (e.key === 'Escape') { closeList(); }
  });
  input.addEventListener('blur', () => setTimeout(closeList, 120));
})();
