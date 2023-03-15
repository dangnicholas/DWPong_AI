import {
    Scene,
    PerspectiveCamera,
    WebGLRenderer,
    AmbientLight,
} from '/vendor/three/build/three.module.js';
import {
    GLTFLoader
} from '/vendor/three/examples/jsm/loaders/GLTFLoader.js';
import {
    GUI
} from '/vendor/three/examples/jsm/libs/dat.gui.module.js';


let container, camera, scene, renderer, opponent;

const morphTargets = {
    Happy: "Happy",
    Frustrated: "Frustrated",
    Mad: "Mad",
    Smug: "Smug",
    Sad: "Sad",
    Right: "Right",
    Left: "Left",
    Confused: "Confused",
}

init();

function init() {
    scene = new Scene();
    console.log("current scene position:")
    //scene.position = window.innerWidth / 2,window.innerHeight /2,0;
    console.log(scene.position);
    //scene.setPosition() 

    camera = new PerspectiveCamera(75, document.body.clientWidth / document.body.clientHeight, 0.1, 1000); //PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000); // LW 1
    camera.position.z = 2; //2;

    scene.add(new AmbientLight(0x8FBCD4, 0.4));

    renderer = new WebGLRenderer({
        alpha: true,
        antialias: true,
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(document.body.clientWidth, document.body.clientHeight);


    //renderer = 500; // LW

    renderer.setAnimationLoop(function () {
        renderer.render(scene, camera);
    });

    container = document.getElementById("opponent");

    container.appendChild(renderer.domElement);

    window.addEventListener('resize', onWindowResize, false);

    console.log("logging from opponent:")
    window.morphOp();
    window.morphOp = morph2;
    window.emptyAnimateFunction();
    window.emptyAnimateFunction = move;
    loadOpponent(scene);
}

async function loadOpponent(scene) {
    const loader = new GLTFLoader();
    const loadedData = await loader.loadAsync('AIOpponent.glb');
    console.log("loaded scene: ")
    console.log(loadedData.scene)
    opponent = loadedData.scene.children[2]; // data contain light and camera object - we can fix this in source at some point, 3rd object is actual mesh
    console.log(opponent);
    scene.add(opponent);

    console.log("current opponent position:")
    console.log(opponent.position);
    opponent.position.x = 1.65;
    opponent.position.y = -0.7;
    opponent.position.z = 0;
    console.log("current opponent position:")
    console.log(opponent.position);

    morph(opponent, morphTargets.Happy, 0.0);

    //initGui();
}

function move(value) {
    opponent.position.y = value;
}

function initGui() {
    const params = {
        Happy: 0,
        Frustrated: 0,
        Mad: 0,
        Smug: 0,
        Sad: 0,
        Right: 0,
        Left: 0,
        Confused: 0,
    }

    const gui = new GUI();
    const folder = gui.addFolder('Morph Targets');

    for (let key in morphTargets) {
        folder.add(params, key, 0, 1).step(0.01).onChange(function (value) {
            morph(opponent, key, value);
        });
    }

}

function morph2(stringE, value) {
    //("morph2!");
    if (value > 1) {
        value = 1
    }
    //console.log(opponent)
    morph(opponent, stringE, value)
}

function morph(mesh, target, value) {
    //console.log(target);
    mesh.morphTargetInfluences[mesh.morphTargetDictionary[target]] = value;
}

function onWindowResize() {

    camera.aspect = document.body.clientWidth / document.body.clientHeight; // window.innerWidth / (window.innerHeight);
    camera.updateProjectionMatrix();
    renderer.setSize(document.body.clientWidth, document.body.clientHeight); //renderer.setSize(window.innerWidth, window.innerHeight);

}