// Avatar 3D (three.js) desenhado a partir do rig que o main.py publica em /rig.
//
// Estilo "cel shading" (MeshToonMaterial) com contorno (cópia da malha virada do avesso,
// um pouco maior). A cabeça roda em 3D (inclinação, rotação lateral, cima/baixo); os olhos
// fecham e seguem o olhar; sobrancelhas, boca e rubor seguem as expressões; tronco, braços,
// mãos e dedos seguem o esqueleto. Coordenadas como em rig_to_dict: normalizadas pela altura
// da imagem, origem no centro, y para cima. A câmara é ortográfica, por isso o avatar fica
// exatamente onde está o da imagem 2D.
import * as THREE from '/three.module.min.js';

const COLORS = {
  skin: 0xf1c7a1, skinDark: 0xd9a47c, hair: 0x5a3825, shirt: 0x2f6fb5, eye: 0xfbfbf8,
  iris: 0x34507a, pupil: 0x141821, brow: 0x4a2e1e, mouth: 0x6e2230, lip: 0x9a3a46,
  blush: 0xf08f8f, outline: 0x1d2230,
};
const OUTLINE = 0.06;           // espessura do contorno (fração do tamanho da peça)
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

// ------------------------------------------------------------------ cena
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(window.devicePixelRatio || 1);
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
camera.position.set(0, 0, 10);
scene.add(new THREE.AmbientLight(0xffffff, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.8);
sun.position.set(0.6, 1.0, 2.0);
scene.add(sun);

// Três tons de luz (cel shading).
const gradient = new THREE.DataTexture(new Uint8Array([110, 110, 110, 255, 190, 190, 190, 255, 255, 255, 255, 255]), 3, 1);
gradient.minFilter = gradient.magFilter = THREE.NearestFilter;
gradient.needsUpdate = true;

/** Materiais de uma parte (cada parte tem os seus, para desvanecer sozinha). */
function palette() {
  const m = {};
  for (const [k, c] of Object.entries(COLORS)) {
    m[k] = k === 'outline'
      ? new THREE.MeshBasicMaterial({ color: c, side: THREE.BackSide, transparent: true })
      : new THREE.MeshToonMaterial({ color: c, gradientMap: gradient, transparent: true });
  }
  m.blush = new THREE.MeshBasicMaterial({ color: COLORS.blush, transparent: true, opacity: 0, depthWrite: false });
  return m;
}
function setOpacity(m, a) {
  for (const [k, mat] of Object.entries(m)) if (k !== 'blush') mat.opacity = a;
}

function mesh(geo, mat, parent, outline = null) {
  const o = new THREE.Mesh(geo, mat);
  if (outline) {
    const line = new THREE.Mesh(geo, outline);
    line.scale.setScalar(1 + OUTLINE);
    o.add(line);
  }
  parent.add(o);
  return o;
}

const SPHERE = new THREE.SphereGeometry(1, 32, 20);
const CYL = new THREE.CylinderGeometry(1, 1, 1, 20);
const UP = new THREE.Vector3(0, 1, 0);
const v3 = (p, z = 0) => new THREE.Vector3(p[0], p[1], z);

/** Cilindro entre a e b (Vector3) com raio r; o contorno só engrossa de lado. */
function placeLimb(m, a, b, r) {
  const d = new THREE.Vector3().subVectors(b, a);
  const len = d.length();
  m.position.copy(a).addScaledVector(d, 0.5);
  if (len > 1e-6) m.quaternion.setFromUnitVectors(UP, d.multiplyScalar(1 / len));
  m.scale.set(r, Math.max(len, 1e-4), r);
  if (m.children[0]) m.children[0].scale.set(1 + 2 * OUTLINE, 1, 1 + 2 * OUTLINE);
}
function placeBall(m, p, r) {
  m.position.copy(p);
  m.scale.setScalar(r);
}

// ------------------------------------------------------------------ cabeça
// Construída com a largura da cara = 1 e depois escalada pelo tamanho do rig.
const hm = palette();
const head = new THREE.Group();
scene.add(head);
const skull = mesh(SPHERE, hm.skin, head, hm.outline);
skull.scale.set(0.5, 0.58, 0.48);
for (const s of [-1, 1]) {
  const ear = mesh(SPHERE, hm.skin, head, hm.outline);
  ear.position.set(s * 0.49, 0.0, -0.04);
  ear.scale.set(0.06, 0.11, 0.08);
}
const hairGroup = new THREE.Group();
head.add(hairGroup);
const hairGeo = new THREE.SphereGeometry(1, 40, 20, 0, Math.PI * 2, 0, Math.PI * 0.45);
const hair = mesh(hairGeo, hm.hair, hairGroup, hm.outline);
hair.scale.set(0.53, 0.61, 0.52);
hair.rotation.x = -0.45;   // a frente sobe (testa à vista), a parte de trás desce
hair.position.y = 0.03;
const backHair = mesh(SPHERE, hm.hair, hairGroup, hm.outline);  // volume atrás da cabeça
backHair.scale.set(0.52, 0.56, 0.42);
backHair.position.set(0, 0.04, -0.1);
const nose = mesh(SPHERE, hm.skinDark, head);
nose.position.set(0, -0.07, 0.47);
nose.scale.set(0.045, 0.05, 0.06);

const eyes = [-1, 1].map((s) => {
  const g = new THREE.Group();
  g.position.set(s * 0.18, 0.05, 0.4);
  head.add(g);
  const white = mesh(SPHERE, hm.eye, g, hm.outline);
  white.scale.set(0.1, 0.09, 0.05);
  const iris = new THREE.Group();
  g.add(iris);
  mesh(SPHERE, hm.iris, iris).scale.set(0.055, 0.06, 0.03);
  const pupil = mesh(SPHERE, hm.pupil, iris);
  pupil.scale.set(0.028, 0.032, 0.02);
  pupil.position.z = 0.015;
  const glint = mesh(SPHERE, hm.eye, iris);
  glint.scale.setScalar(0.014);
  glint.position.set(0.02, 0.025, 0.03);
  iris.position.z = 0.03;
  const brow = mesh(new THREE.BoxGeometry(1, 1, 1), hm.brow, head);
  brow.scale.set(0.17, 0.032, 0.03);
  return { g, iris, brow, side: s };
});

const mouth = new THREE.Group();
mouth.position.set(0, -0.26, 0.43);
head.add(mouth);
const mouthInside = mesh(SPHERE, hm.mouth, mouth);
const lip = mesh(new THREE.BufferGeometry(), hm.lip, mouth);
let lipKey = '';
const blush = [-1, 1].map((s) => {
  const b = mesh(new THREE.CircleGeometry(1, 24), hm.blush, head);
  b.position.set(s * 0.26, -0.12, 0.43);
  b.rotation.y = s * 0.55;
  b.scale.setScalar(0.07);
  return b;
});

function updateMouth(open, smile) {
  const hw = 0.1 * (1 + 0.35 * smile);
  const h = 0.01 + 0.13 * open;
  mouthInside.scale.set(hw * 0.95, h, 0.04);
  mouthInside.position.y = -h * 0.55;
  mouthInside.visible = open > 0.06;
  // Linha da boca: curva com os cantos a subir quando sorri. Só refaz a geometria quando muda.
  const key = `${hw.toFixed(3)}|${smile.toFixed(2)}|${open.toFixed(2)}`;
  if (key === lipKey) return;
  lipKey = key;
  const lift = 0.06 * smile;
  const curve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-hw, lift, 0), new THREE.Vector3(0, -lift - 0.01, 0.02), new THREE.Vector3(hw, lift, 0));
  lip.geometry.dispose();
  lip.geometry = new THREE.TubeGeometry(curve, 16, 0.012, 6, false);
}

function updateHead(d) {
  const h = d.head;
  head.visible = !!h;
  if (!h) return;
  head.position.copy(v3(h.center, 0.4));
  head.scale.setScalar(h.size);
  // A rotação lateral vem do "turn" (para onde aponta o nariz na imagem), mais fiável do que o yaw.
  head.rotation.set(THREE.MathUtils.clamp(h.pitch, -0.6, 0.6), THREE.MathUtils.clamp(h.turn * 0.8, -0.9, 0.9),
    h.roll, 'YXZ');
  setOpacity(hm, h.alpha);
  for (const [i, e] of eyes.entries()) {
    e.g.scale.y = THREE.MathUtils.clamp(h.eye_open[i], 0.07, 1.3);
    e.iris.position.x = h.look[0] * 0.035;
    e.iris.position.y = h.look[1] * 0.03;
    e.brow.position.set(e.side * 0.18, 0.19 + h.brow * 0.05, 0.44);
    // Franzir baixa as pontas de dentro; levantar sobe-as um pouco.
    e.brow.rotation.z = e.side * -h.brow * 0.3;
  }
  updateMouth(h.mouth_open, h.mouth_smile);
  for (const b of blush) b.material.opacity = THREE.MathUtils.clamp((h.mouth_smile - 0.3) / 0.4, 0, 1) * 0.55 * h.alpha;
  // Movimento secundário do cabelo (desvio em unidades da cabeça).
  hairGroup.position.set(d.hair[0] / h.size, d.hair[1] / h.size, 0);
  hairGroup.rotation.z = d.hair_rot;
}

// ------------------------------------------------------------------ corpo
const bm = palette();
const body = new THREE.Group();
scene.add(body);
const torso = mesh(new THREE.CylinderGeometry(0.5, 0.42, 1, 32), bm.shirt, body, bm.outline);
const chest = mesh(SPHERE, bm.shirt, body, bm.outline);  // topo arredondado do tronco
const shoulderCaps = [0, 1].map(() => mesh(SPHERE, bm.shirt, body, bm.outline));
const neck = mesh(CYL, bm.skin, body, bm.outline);

function updateBody(d) {
  const b = d.body;
  body.visible = !!b;
  if (!b) return;
  const sw = b.width;
  const [ls, rs] = b.shoulders.map((p) => v3(p));
  const top = ls.clone().add(rs).multiplyScalar(0.5);
  const bottom = v3(b.hips[0]).add(v3(b.hips[1])).multiplyScalar(0.5);
  const axis = new THREE.Vector3().subVectors(bottom, top).normalize();
  const start = top.clone().addScaledVector(axis, -0.12 * sw);
  placeLimb(torso, start, bottom, 1);
  torso.scale.x = 1.2 * sw;
  torso.scale.z = 0.55 * sw;
  torso.children[0].scale.set(1 + OUTLINE / 2, 1, 1 + OUTLINE);
  chest.position.copy(start).addScaledVector(axis, 0.1 * sw);
  chest.quaternion.copy(torso.quaternion);
  chest.scale.set(0.6 * sw, 0.16 * sw, 0.275 * sw);
  for (const [i, p] of [ls, rs].entries()) placeBall(shoulderCaps[i], p.clone().setZ(0.02), 0.17 * sw);
  const neckTop = d.head ? v3(d.head.center).addScaledVector(UP, -0.3 * d.head.size) : v3(b.neck_base).addScaledVector(UP, 0.25 * sw);
  placeLimb(neck, v3(b.neck_base, 0.05), neckTop.setZ(0.15), 0.09 * sw);
}

// ------------------------------------------------------------------ braços e mãos
const arms = {};
const hands = {};
function arm(side) {
  if (!arms[side]) {
    const m = palette();
    const g = new THREE.Group();
    scene.add(g);
    arms[side] = {
      g, upper: mesh(CYL, m.shirt, g, m.outline), elbow: mesh(SPHERE, m.shirt, g, m.outline),
      fore: mesh(CYL, m.skin, g, m.outline), wrist: mesh(SPHERE, m.skin, g, m.outline),
    };
  }
  return arms[side];
}
function hand(side) {
  if (!hands[side]) {
    const m = palette();
    const g = new THREE.Group();
    scene.add(g);
    const palm = mesh(SPHERE, m.skin, g, m.outline);
    const fingers = FINGERS.map((f) => ({
      segs: f.slice(1).map(() => mesh(CYL, m.skin, g, m.outline)),
      joints: f.map(() => mesh(SPHERE, m.skin, g, m.outline)),
    }));
    hands[side] = { g, m, palm, fingers };
  }
  return hands[side];
}

function updateArms(d) {
  const sw = d.body ? d.body.width : 0.5;
  for (const side of ['15', '16']) {
    const a = d.arms[side];
    const A = arm(side);
    A.g.visible = !!a;
    if (!a) continue;
    const z = 0.7;
    const s = v3(a.shoulder, z), e = v3(a.elbow, z), w = v3(a.wrist, z);
    placeLimb(A.upper, s, e, 0.11 * sw);
    placeBall(A.elbow, e, 0.11 * sw);
    placeLimb(A.fore, e, w, 0.085 * sw);
    placeBall(A.wrist, w, 0.085 * sw);
  }
  for (const side of ['15', '16']) {
    const h = d.hands[side];
    const H = hand(side);
    H.g.visible = !!h;
    if (!h) continue;
    setOpacity(H.m, h.alpha);
    const p = h.points.map((q) => v3(q, 0.85));
    const width = p[5].distanceTo(p[17]) || 0.05;
    const r = 0.17 * width;
    const center = [0, 5, 9, 13, 17].reduce((acc, i) => acc.add(p[i]), new THREE.Vector3()).multiplyScalar(1 / 5);
    placeBall(H.palm, center, 0.62 * width);
    H.palm.scale.z = 0.3 * width;
    for (const [fi, f] of FINGERS.entries()) {
      const F = H.fingers[fi];
      const k = fi === 0 ? 1.15 : 1.0;  // polegar um pouco mais grosso
      for (let j = 0; j < 3; j++) placeLimb(F.segs[j], p[f[j]], p[f[j + 1]], r * k * (1 - 0.1 * j));
      for (let j = 0; j < 4; j++) placeBall(F.joints[j], p[f[j]], r * k * (1 - 0.1 * Math.max(0, j - 1)));
    }
  }
}

// ------------------------------------------------------------------ ciclo
let frameAspect = 16 / 9;
function fit() {
  const W = window.innerWidth, H = window.innerHeight, win = W / H;
  renderer.setSize(W, H);
  let halfW = frameAspect, halfH = 1;
  if (win > frameAspect) halfW = win; else halfH = frameAspect / win;
  Object.assign(camera, { left: -halfW, right: halfW, top: halfH, bottom: -halfH });
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', fit);
fit();

function update(d) {
  if (d.aspect !== frameAspect) { frameAspect = d.aspect; fit(); }
  updateHead(d);
  updateBody(d);
  updateArms(d);
  renderer.render(scene, camera);
}

let seq = -1;
window.avatar3d = { update, scene, frames: 0 };  // para testes e depuração
async function loop() {
  try {
    const r = await fetch('/rig?after=' + seq, { cache: 'no-store' });
    if (r.status === 200) {
      seq = +r.headers.get('X-Seq');
      update(await r.json());
      window.avatar3d.frames++;
    }
  } catch (e) {
    await new Promise((res) => setTimeout(res, 500));
  }
  // setTimeout e não requestAnimationFrame (pára com a fonte escondida no OBS); o ritmo vem
  // do servidor (/rig só responde quando há um rig novo).
  setTimeout(loop, 0);
}
head.visible = body.visible = false;  // até chegar o primeiro rig
renderer.render(scene, camera);
loop();
