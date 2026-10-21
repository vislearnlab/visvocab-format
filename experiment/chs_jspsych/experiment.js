/*
 * Visual format study (photos vs cut-outs) for the CHS jsPsych runner.
 * Paste this whole file into "jsPsych Experiment Code" on the CHS study design page.
 *
 * Adapted from the visvocab CHS Lookit protocol (experiment/lookit_reference/experiment_sample2.js):
 * same page colour, trial length, carrier phrases, attention-getters, calibration, consent and debrief text.
 * What changed: hard pairs only; every pair is shown twice, once as photos and once as cut-outs,
 * in opposite halves of the session, with the other word named the second time; four
 * counterbalancing lists; Lookit frames replaced by jsPsych 8 plugins that CHS loads by default
 * (html-button-response, video-button-response, preload, fullscreen) plus chsRecord/chsSurvey.
 *
 * CHS provides initJsPsych, the jsPsych plugins, chsRecord, chsSurvey and window.chs.child.
 * preview.html provides stand-ins for the CHS parts so this file runs locally.
 */

const BASE = "https://ucsdlearninglabs.org/stimuli/";
const IMG_BASE = window.VF_IMG_BASE || BASE + "visvocab-format/";   // photo/{word}.png, cutout/{word}.png (upload stimuli/images here)
const TRIAL_MS = 7200;                         // as CHS; audio files start with ~2 s of silence (silent preview), carrier ~2.0-3.1 s
const ATTENTION_MS = 5100;
const CARRIERS = ["where", "find", "look_at", "see"];

const PAIRS = [
  ["cloud", "tree"], ["squirrel", "monkey"], ["turkey", "goat"], ["duck", "chicken"],
  ["balloon", "kite"], ["truck", "car"], ["lizard", "snake"], ["turtle", "frog"],
  ["crow", "penguin"], ["camel", "elephant"], ["snail", "worm"], ["bulldozer", "tractor"],
  ["acorn", "coconut"], ["cow", "horse"], ["butterfly", "ladybug"], ["rabbit", "hamster"],
];
const ATTENTION_GETTERS = ["Balloons", "BouncyBalls", "Meadow", "Mountains"];

/* ---------------- trial list ---------------- */

function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/*
 * List L (0-3) puts pair i in cell (i + L) % 4, where a cell fixes which format comes first
 * and which word is named first. Across the four lists every pair visits every cell once, so each
 * word is named in each format equally often; within a list, 8 pairs are photo-first and 8 cut-out-first.
 */
function makeTrials(list) {
  const half1 = [], half2 = [];
  PAIRS.forEach(([a, b], i) => {
    const cell = (i + list) % 4;
    const firstFormat = cell % 2 === 0 ? "photo" : "cutout";
    const secondFormat = firstFormat === "photo" ? "cutout" : "photo";
    const [firstNamed, secondNamed] = cell < 2 ? [a, b] : [b, a];
    const pair = a + "-" + b;
    half1.push({ pair, format: firstFormat, target: firstNamed, distractor: firstNamed === a ? b : a, half: 1 });
    half2.push({ pair, format: secondFormat, target: secondNamed, distractor: secondNamed === a ? b : a, half: 2 });
  });
  let h1 = shuffle(half1), h2 = shuffle(half2);
  // the same pair must not appear back to back across the half boundary
  while (h2[0].pair === h1[h1.length - 1].pair) h2 = shuffle(half2);
  const trials = h1.concat(h2);
  let carriers;
  trials.forEach((t, k) => {
    if (k % 4 === 0) carriers = shuffle(CARRIERS);   // each carrier once per block of four, as in CHS
    t.carrier = carriers[k % 4];
    t.target_side = Math.random() < 0.5 ? "left" : "right";
    t.trial_index_in_list = k;
    t.list = list;
  });
  return trials;
}

/* ---------------- session setup ---------------- */

const params = new URLSearchParams(window.location.search);
const LIST = params.has("list") ? Number(params.get("list")) % 4 : Math.floor(Math.random() * 4);
const TRIALS = makeTrials(LIST);

let childAgeMonths = 20;   // default as in CHS: older CDI form if age is unavailable
try {
  const bday = window.chs.child.attributes.birthday;
  childAgeMonths = (new Date() - new Date(bday)) / (1000 * 60 * 60 * 24 * 30.43) || 20;
} catch (e) { console.error(e); }
let childId = "";
try { childId = window.chs.child.id; } catch (e) { console.error(e); }
const CDI_LINK = childAgeMonths <= 18
  ? `https://ucsd.co1.qualtrics.com/jfe/form/SV_3UTxIUSEYbe3QdE?ChildID=${childId}`
  : `https://qualtricsxmhcv2m8q6d.qualtrics.com/jfe/form/SV_7NAZ6riQu1aBIZU?ChildID=${childId}`;

// use_webaudio: false -> HTML5 <audio>, as the Lookit frames used. ucsdlearninglabs.org/stimuli sends no
// Access-Control-Allow-Origin header, so WebAudio's fetch (and preloading video by XHR) is blocked from CHS.
const jsPsych = initJsPsych({
  use_webaudio: false,
  extensions: [{ type: chsRecord.TrialRecordExtension, params: { wait_for_upload_message: "", max_upload_seconds: 10 } }],
});
jsPsych.data.addProperties({ list: LIST, study: "visvocab-format" });

const audio = (name) => BASE + "mp3/" + name + ".mp3";
const img = (format, word) => IMG_BASE + format + "/" + word + ".png";

/* ---------------- shared pieces ---------------- */

// CSS gray page (128); each 600 px image is 37.5% of a 1600 px-wide screen with a 12.5% gap, as in the CHS layout.
const GRAY_PAGE = "<style>body, .jspsych-display-element { background: gray !important; cursor: none; }" +
  ".vf-pair { display: flex; justify-content: center; gap: 12.5vw; } .vf-img { width: 37.5vw; max-height: 85vh; object-fit: contain; }</style>";
const CLEAR_PAGE = "<style>body, .jspsych-display-element { background: white; cursor: auto; }</style>";

let paused = false;   // set by the space bar during any looking trial
function listenForPause() {
  paused = false;
  return jsPsych.pluginAPI.getKeyboardResponse({
    callback_function: () => { paused = true; jsPsych.pluginAPI.clearAllTimeouts(); jsPsych.finishTrial({ paused: true }); },
    valid_responses: [" "], rt_method: "performance", persist: false, allow_held_key: false,
  });
}

// Shown only after a space-bar pause; the paused trial is skipped, as with frameOffsetAfterPause: 1 in CHS.
const pauseScreen = {
  timeline: [{
    type: jsPsychHtmlButtonResponse,
    stimulus: CLEAR_PAGE + "<p style='font-size:22px'>The study is paused.</p><p>Please reposition your baby and press the space bar to continue.</p>",
    choices: ["Continue"],
    on_load: () => jsPsych.pluginAPI.getKeyboardResponse({ callback_function: () => jsPsych.finishTrial({ resumed_by: "space" }), valid_responses: [" "], persist: false }),
    data: { trial_type_name: "pause" },
  }],
  conditional_function: () => paused,
};

function playAudio(src, data) {
  jsPsych.pluginAPI.getAudioPlayer(src).then((player) => {
    data.audio_onset = performance.now();
    player.play();
  }).catch((e) => console.error("audio", src, e));
}

/* ---------------- looking trials ---------------- */

function lookingTrial(t) {
  const left = t.target_side === "left" ? t.target : t.distractor;
  const right = t.target_side === "left" ? t.distractor : t.target;
  const onset = {};
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: GRAY_PAGE +
      `<div class="vf-pair"><img class="vf-img" src="${img(t.format, left)}"><img class="vf-img" src="${img(t.format, right)}"></div>`,
    choices: [],
    trial_duration: TRIAL_MS,
    response_ends_trial: false,
    extensions: [{ type: chsRecord.TrialRecordExtension }],
    data: Object.assign({ trial_type_name: "test", left_image: left, right_image: right,
                          audio_file: t.carrier + "_" + t.target }, t),
    on_load: () => {
      onset.image_onset = performance.now();
      playAudio(audio(t.carrier + "_" + t.target), onset);
      listenForPause();
    },
    on_finish: (data) => {
      jsPsych.pluginAPI.cancelAllKeyboardResponses();
      Object.assign(data, onset, { paused: paused });
    },
  };
}

function attentionGetter(name) {
  return {
    type: jsPsychVideoButtonResponse,
    stimulus: [BASE + "mp4/" + name + ".mp4"],
    choices: [],
    width: Math.round(window.innerWidth * 0.5),
    trial_duration: ATTENTION_MS,
    trial_ends_after_video: false,
    response_ends_trial: false,
    autoplay: true,
    prompt: GRAY_PAGE,
    data: { trial_type_name: "attention_getter", video: name },
    on_load: listenForPause,
    on_finish: () => jsPsych.pluginAPI.cancelAllKeyboardResponses(),
  };
}

// Test trials with an attention-getter after every fourth one (not at the very end), as in CHS.
function testTimeline() {
  const ags = shuffle(ATTENTION_GETTERS.concat(ATTENTION_GETTERS));
  const tl = [];
  TRIALS.forEach((t, k) => {
    tl.push(lookingTrial(t), pauseScreen);
    if ((k + 1) % 4 === 0 && k < TRIALS.length - 1) tl.push(attentionGetter(ags[((k + 1) / 4 - 1) % ags.length]), pauseScreen);
  });
  return tl;
}

/* ---------------- calibration (CHS exp-lookit-calibration: centre, left, right, centre; 3 s each) ---------------- */

const CALIB_POS = { center: "50%", left: "12%", right: "88%" };
const calibration = ["center", "left", "right", "center"].map((pos) => ({
  type: jsPsychHtmlButtonResponse,
  stimulus: GRAY_PAGE + `<style>@keyframes vfbounce { 0%,100% { transform: translate(-50%,-50%) } 50% { transform: translate(-50%,-65%) } }</style>
    <img src="${BASE}img/calib_duck.jpg" style="position:fixed; top:50%; left:${CALIB_POS[pos]}; width:12vw; transform:translate(-50%,-50%); animation: vfbounce .6s infinite;">`,
  choices: [],
  trial_duration: 3000,
  response_ends_trial: false,
  extensions: [{ type: chsRecord.TrialRecordExtension }],
  data: { trial_type_name: "calibration", position: pos },
  on_load: () => playAudio(audio("pinwheel"), {}),
}));

/* ---------------- parent-facing screens ---------------- */

const eligibility = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<h3>Eligibility survey</h3><img src="${BASE}img/USA_map.jpg" alt="US map" style="max-width:420px"><p>Do you (the grown-up) and your baby live in the United States of America (USA)?</p>`,
  choices: ["Yes", "No"],
  data: { trial_type_name: "eligibility" },
};
const notEligible = {
  timeline: [{
    type: jsPsychHtmlButtonResponse,
    stimulus: `<h3>This study is only open to families in the United States.</h3><p style="max-width:700px; margin:auto">Thank you so much for your interest in our study! Unfortunately, this study is only open to families in the United States due to difficulties with compensating families in other countries. If you live outside of the United States, <a href="https://lookit.mit.edu/studies/">please click here to exit and go back to the studies page.</a> If you choose to participate and you live outside of the United States, we won't be able to use your data or compensate you for your time.</p>`,
    choices: ["Continue anyway"],
  }],
  conditional_function: () => jsPsych.data.get().filter({ trial_type_name: "eligibility" }).last(1).values()[0].response === 1,
};

function instructionVideo(video, intro, summary, button) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `<h3>Study instructions</h3><p>${intro}</p>
      <div style="display:flex; gap:24px; text-align:left; max-width:1100px; margin:auto">
        <video src="${BASE}mp4/${video}.mp4" controls style="width:55%"></video>
        <div style="width:45%; font-size:15px">${summary}</div></div>`,
    choices: [button],
  };
}

const intro = instructionVideo("instructions1",
  "Welcome to the study! Please watch this video to get started. (Or you can read the summary to the right if you prefer.)",
  `<b>Background information about the study</b><ul><li>Your baby does not need to be with you at this point in the study. We will let you know when it is time to get your baby.</li>
   <li>Over the first two years of life, babies learn to link the words they hear with the objects they experience in their environment. In this study, we want to understand how babies recognize familiar objects in different kinds of pictures.</li></ul>
   <b>Preview of what your baby will see</b><ul><li>Your baby will be shown two objects on the screen, one on the left and one on the right, and will hear one of them named. Sometimes the objects are shown in a full photograph and sometimes on a plain grey background.</li></ul>
   <b>What's next?</b><ul><li>We will check that your webcam is set up so we can record your baby's looking, then ask you to record a short video giving consent.</li></ul>`,
  "I'm ready to make sure my webcam is connected!");

const howTo = instructionVideo("instructions2",
  "Let's get ready! Please watch this video to learn what to do during the study. (Or you can read the summary to the right if you prefer.)",
  `<b>What to do during the study</b><ul><li>If your child gets fussy or distracted, or you need to attend to something else, you can pause the study by pressing the space bar.</li>
   <li>If your baby becomes bored or too fussy to finish, that is perfectly OK. You can stop at any point; we can still use the parts your baby watched.</li></ul>
   <b>Your role</b><ul><li>Keep your baby seated on your lap and facing the screen so we can see their eyes, and so we cannot see your face.</li></ul>
   <b>Please close your eyes during the study</b><ul><li>Please keep your eyes closed and don't talk to your baby or point. Move the cursor off the screen. You will hear an audio prompt when the study is over. It lasts about 5 minutes.</li></ul>`,
  "Getting in position");

const soundCheck = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<h3>Holding your baby and sound check</h3>
    <img src="${BASE}img/placeholder_lap_image.jpg" style="max-width:380px"><p style="max-width:700px; margin:auto">Please sit facing the monitor, holding your child on your lap, with the webcam angled so that your child's eyes are visible and your eyes are not.</p>
    <p>Play the sample audio and adjust your volume so it sounds like a person's voice in the room. You should hear "Ready to go?"</p>
    <audio controls src="${BASE}mp3/ready_to_go.mp3"></audio>`,
  choices: ["Next"],
};

const consent = {
  type: chsRecord.VideoConsentPlugin,
  template: "consent-template-5",
  PIName: "Bria Long",
  PIContact: "Visual Learning Lab at vislearnlab@ucsd.edu",
  institution: "University of California, San Diego",
  purpose: "We are running this study to better understand how children develop their language and category representations. This research study aims to to understand how differences in children’s language learning are related to how they learn categories from their everyday visual experiences. We will investigate how they represent and interpret different categories and objects across age and interact with different types and forms of objects.",
  procedures: "During the study, your child will play different games and activities. The activities will examine your child's linguistic, categorization, or cognitive skills. In one game, your child may play a picture matching game where your child matches a word heard with one of up to 4 images on a screen. In other games, for example, your child may play a ball-identifying game where they identify the box a ball is placed in after the boxes are moved around, or an object-labeling game where your child sees an object or a drawing on the screen and then matches them with one of up to 4 new objects on the screen. All objects and words used in this study will be age-appropriate. Children will identify their choices in these games either by looking at it, pointing at it, or indicating their choice to a parent who then performs one of the actions. The alternative to being in this study is not to participate.",
  voluntary_participation: "Refusal to participate will involve no penalty or loss of compensation. Further, if you decide to stop participating, you can email us and we can remove any information collected about you and your child if you would like us to. You can also email us at any point after the study to withdraw permission and any information collected about you and your child will be deleted and not used for future research. Any deletion of information will not be held against you.",
  risk_statement: "There are minimal risks involved with this study. The main risk is a loss of confidentiality. The other associated risks are mild boredom or anxiety about task performance. Should the child participant experience mild boredom or anxiety, we do not believe this will leave a lasting impact.",
  payment: "We cannot promise any benefit to you or to others from you participating in this research. However, possible benefits may include enjoyment from engaging in the study because of the stimulating nature of the experimental games. Parents may also benefit from the satisfaction of having contributed to research on development in early childhood. There are also potential benefits to educational and public policy regarding the benefits of supporting language development in diverse populations and in children who struggle with early category learning.",
  datause: "Study information stored by researchers will be labeled with a subject code instead of your name or other information that can easily identify you, which will be kept separate from the rest of the study information. While we cannot guarantee complete confidentiality, we will limit access to information, documents, or recordings of you and your child. Only people who have a need to review your information, documents, or recordings will have access. These people might include:<br><ul><li>Members of the research team and other staff or representatives of UCSD whose work is related to the research or to protecting your rights and safety.</li><li>Representatives of the study sponsor or product manufacturer</li><li>Representatives of Federal and other regulatory agencies who make sure the study is done properly and that your rights and safety are protected.</li></ul>According to the Lookit Terms of Use, we may download local copies of your Lookit data, including study data and video, and account, child, and demographic survey data from participants who have participated in their studies. Lookit may also keep copies of this data, including study data and video, indefinitely. Our research team also retains ownership of study data, including video, for our own studies. You cannot take part in this study without agreeing to the Lookit Terms of Use.",
  additional_video_privacy_statement: "These video recordings will include your and your child’s faces and/or parts of your or your child’s bodies. While these videos will be subject to the confidentiality conditions described in this form, it is possible that someone who knows you well may be able to identify you from the photos and know you are participating in this study. To minimize this possibility, your videos will only be shared within the research team and people who have a need to review your recordings listed above. It will not be used in publications or for external research unless you explicitly provide permission to do so. Further, if you would not like us to use your recordings at any point in the future, you can email us to withdraw permission and all recordings of your child and you will be deleted and not used for future research.",
  gdpr: false,
  research_rights_statement: "If before or during your participation in the study you have questions about your rights as a research participant, or you want to talk to someone outside the research team, please contact UC San Diego Office of IRB Administration at 858-246-4777 or irb@ucsd.edu",
  additional_segments: [{
    title: "Will I be compensated for participating in the research?",
    text: "You will receive $5 for completing this study for your time and effort as is standard on Children Helping Science. You will only receive compensation once per child and provided your child is visible at some point and in the age range specified by the study. Compensation will take the form of an Amazon gift card, typically sent to you electronically within a week. Compensation does not depend on finishing the whole study and you can withdraw from the study at any point without affecting your compensation.",
  }],
};

const getReady = {
  type: jsPsychHtmlButtonResponse,
  stimulus: GRAY_PAGE + `<h2>Get ready!</h2><p>Please close your eyes!</p><img src="${BASE}img/calib_duck.jpg" style="width:20vw">`,
  choices: [],
  trial_duration: 3000,
  on_load: () => playAudio(audio("get_ready"), {}),
};

const greatJob = {
  type: jsPsychHtmlButtonResponse,
  stimulus: GRAY_PAGE + `<img src="${BASE}img/calib_duck.jpg" style="width:20vw">`,
  choices: [],
  trial_duration: 3000,
  on_load: () => playAudio(audio("great_job"), {}),
};

const vocabSurvey = {
  type: jsPsychHtmlButtonResponse,
  stimulus: CLEAR_PAGE + `<h3>Vocabulary survey</h3><p style="max-width:700px; margin:auto">You are almost done with the study! Please additionally complete <a href="${CDI_LINK}" target="_blank" rel="noopener">this survey</a> where we will ask you a few questions about what words your child can understand and can produce.</p>`,
  choices: ["Next"],
};

const debrief = {
  type: jsPsychHtmlButtonResponse,
  stimulus: `<h3>Thank you!</h3><img src="${BASE}img/exit_survey_image.jpg" alt="thank you" style="max-width:300px">
    <div style="max-width:760px; margin:auto; text-align:left">
    <p>You are done with the study! After you exit this experiment, we will check that your consent video meets our eligibility criteria and send you a $5 Amazon gift card (US) within one week of participating.</p>
    <p>In this study, we are interested in how babies recognize the objects that words refer to, and whether the background of a picture helps or distracts them. Your baby saw pairs of similar objects, such as a cow and a horse, and heard one of them named. Each pair was shown twice: once as a full photograph and once with the background replaced by plain grey. The order of the pictures, the word that was named, and the side of the screen were all randomized.</p>
    <p>Babies know more about words than they are able to say, so we look at what babies <i>understand</i> by seeing what they look at. A background can help, because a cow is usually in a field, or it can distract, by drawing your baby's eyes away from the object. Comparing the two kinds of pictures tells us which. You can <a href="https://www.vislearnlab.org/">visit our website</a> to learn more about our research!</p></div>`,
  choices: ["Finish"],
};

/* ---------------- run ---------------- */

const preload = {
  type: jsPsychPreload,
  images: TRIALS.flatMap((t) => [img(t.format, t.target), img(t.format, t.distractor)]).concat([BASE + "img/calib_duck.jpg"]),
  audio: TRIALS.map((t) => audio(t.carrier + "_" + t.target)).concat(["pinwheel", "get_ready", "great_job"].map(audio)),
  // attention-getter videos are not preloaded (XHR is blocked by CORS); <video> streams them
  message: "Loading the pictures and sounds...",
  max_load_time: 120000,
};

const setup = [
  eligibility, notEligible,
  intro,
  { type: chsRecord.VideoConfigPlugin, troubleshooting_intro: "If you’re having any difficulties getting your webcam set up, please feel free to email the Visual Learning Learning Lab at vislearnlab@ucsd.edu." },
  consent,
  howTo, soundCheck,
];
// ?skip=1 starts at the looking task (piloting / preview only; never on a live study link: no consent is recorded)
jsPsych.run([
  ...(params.get("skip") === "1" ? [] : setup),
  preload,
  { type: jsPsychFullscreen, fullscreen_mode: true, message: "<p>The study will switch to full screen mode when you press the button below.</p>" },
  getReady,
  ...calibration,
  ...testTimeline(),
  greatJob,
  { type: jsPsychFullscreen, fullscreen_mode: false },
  vocabSurvey,
  { type: chsSurvey.ExitSurveyPlugin },
  debrief,
]);
