// three.js view of the intake arm, adapted from the anywidget version in arm_viewer.js.
// The anywidget comm is gone: the model is fetched once, and the page calls setAngle()
// directly as the reader scrubs through a run.

import * as THREE from "https://esm.sh/three@0.169.0";
import { GLTFLoader } from "https://esm.sh/three@0.169.0/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "https://esm.sh/three@0.169.0/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "https://esm.sh/three@0.169.0/examples/jsm/environments/RoomEnvironment.js";

// Fitted once from the CAD with glbtools.fit_shaft_axis and baked in here, so the
// browser never needs a mesh library just to find the shaft it pivots about.
export const ARM = {
  glb: "final.glb",
  movingNode: "INTAKE assembly redo redo <1>",
  pivotPoint: [-0.070354, 0.41553, 0.160339],
  pivotAxis: [0, 1, 0],
  // How much room to leave around the model. 1.0 fits its bounding sphere exactly to the
  // vertical field of view; above that is margin.
  framePadding: 1.05,
  // Look down the pivot axis so the arm swings in the plane of the screen.
  lookFrom: [0, -1, 0],
  cameraUp: [0, 0, 1],
  // The CAD was exported with the arm 45 degrees above horizontal, which is where the
  // simulation reads theta = +45. Rotations are applied relative to that pose.
  cadPoseDeg: 45,
};

// three.js rewrites glTF node names (spaces become underscores, '.' ':' '/' '[' ']' are
// dropped) so that they can be used as animation paths. Collapse both sides to compare.
const squash = (s) => s.replace(/[\s._:/[\]]/g, "").toLowerCase();

function findByName(root, name) {
  const want = squash(name);
  let exact = null, partial = null;
  root.traverse((obj) => {
    if (!obj.name || exact) return;
    const got = squash(obj.name);
    if (got === want) exact = obj;
    else if (!partial && want && got.includes(want)) partial = obj;
  });
  return exact || partial;
}

// Every chapter shows the arm, but the 3.4 MB model is fetched and parsed exactly once
// and then cloned. Object3D.clone() shares geometries and materials by reference, so the
// extra viewers cost a scene graph rather than another copy of the mesh data.
let modelPromise = null;

function loadArmModel() {
  if (!modelPromise) {
    modelPromise = fetch(ARM.glb)
      .then((r) => {
        if (!r.ok) throw new Error(`${ARM.glb}: HTTP ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => new Promise((res, rej) => new GLTFLoader().parse(buf, "", res, rej)));
  }
  return modelPromise;
}

export async function createArmViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.domElement.style.cssText = "display:block;width:100%;height:100%";
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  // Machined aluminium and anodized parts look flat under bare lights, so light the
  // scene with an environment map plus one directional light for highlights.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(1, 2, 1.5);
  scene.add(sun);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
  const controls = new OrbitControls(camera, renderer.domElement);

  // Render on demand rather than in a free-running loop: the scene is static except when
  // the reader orbits or the angle changes.
  let pending = null;
  const invalidate = () => {
    if (pending === null) {
      pending = requestAnimationFrame(() => { pending = null; renderer.render(scene, camera); });
    }
  };
  controls.addEventListener("change", invalidate);

  const resize = new ResizeObserver(() => {
    const w = container.clientWidth || 1, h = container.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    invalidate();
  });
  resize.observe(container);

  const root = (await loadArmModel()).scene.clone(true);
  scene.add(root);

  const target = findByName(root, ARM.movingNode);
  if (!target) throw new Error(`no node matching ${ARM.movingNode}`);

  // Frame the whole model rather than just the subtree that moves. The moving subtree is
  // off to one side of the robot, so framing on it alone put the far end of the assembly
  // outside the view; its bounding sphere is also the natural thing to orbit about.
  const sphere = new THREE.Box3().setFromObject(root).getBoundingSphere(new THREE.Sphere());
  const center = sphere.center.clone();
  const r = sphere.radius || 1;
  camera.near = r / 100;
  camera.far = r * 100;
  // OrbitControls derives its azimuth from camera.up, so set it before update().
  camera.up.set(...ARM.cameraUp).normalize();
  camera.position.copy(center).addScaledVector(
    new THREE.Vector3(...ARM.lookFrom).normalize(),
    (r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * ARM.framePadding);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
  // Remember this pose so that the chapter's reset button can come back to it after the
  // reader has orbited away.
  controls.saveState();

  // Insert a group at the pivot between the target and its parent, so rotating the group
  // rotates the whole subtree about the shaft. `attach` reparents while preserving the
  // target's world transform, and the pivot point and axis are converted into the
  // parent's space in case any ancestor carries a transform of its own.
  const parent = target.parent;
  parent.updateWorldMatrix(true, false);
  const toLocal = new THREE.Matrix4().copy(parent.matrixWorld).invert();
  const pivot = new THREE.Group();
  parent.add(pivot);
  pivot.position.copy(new THREE.Vector3(...ARM.pivotPoint).applyMatrix4(toLocal));
  pivot.updateMatrixWorld(true);
  pivot.attach(target);
  const axisLocal = new THREE.Vector3(...ARM.pivotAxis).transformDirection(toLocal).normalize();

  return {
    // theta is measured up from horizontal, while a +alpha rotation about the axis
    // carries the arm down, so the swing is the negation of the change in theta.
    setAngle(thetaDeg) {
      const swing = -(thetaDeg - ARM.cadPoseDeg);
      pivot.quaternion.setFromAxisAngle(axisLocal, THREE.MathUtils.degToRad(swing));
      invalidate();
    },
    // Back to the framing above, undoing whatever orbiting and zooming has happened.
    resetView() {
      controls.reset();
      invalidate();
    },
    dispose() {
      resize.disconnect();
      if (pending !== null) cancelAnimationFrame(pending);
      // Deliberately not disposing geometries or materials: they are shared with the
      // other chapters' viewers, and freeing them here would blank those out.
      pmrem.dispose();
      renderer.dispose();
    },
  };
}
