#include <Wire.h>
#include <math.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define BASE_CENTER 1500 
#define BASE_HGT 130   
#define HUMERUS 161   
#define ULNA 162      
#define GRIPPER 131   
float hum_sq = (float)HUMERUS * (float)HUMERUS;
float uln_sq = (float)ULNA * (float)ULNA;

# calibrated offsets — yours will be different, don't copy these blindly
int baseOffset = -110;       
int shoulderOffset = -60;   
int elbowOffset = 110;      
int wristYawOffset = 60;    // forearm roll
int wristPitchOffset = 55;   
int wristRollOffset = -60;   
int gripperOffset = 100;    

// Vision → world Z in calculate_real_coords() (same mm frame as IK z).
// Arm too low? bump COORDS_Z_OFFSET_MM or measure pivot_Z / lens_up better.
// Distance off? mess with COORDS_DIST_SCALE.
static const float COORDS_DIST_SCALE = 1.0f;
static const float COORDS_Z_OFFSET_MM = 0.0f;
// pitch→Z was undershooting height changes, so we juice it a bit
static const float COORDS_Z_GAIN = 2.2f;
// ToF is behind the tip a little — subtract so the gripper actually hits the object.
// stops short -> decrease this. overshoots -> increase.
static const float COORDS_TOF_TO_GRIPPER_TIP_MM = 25.0f;
// if "object higher -> arm goes lower" flip this / pitch sign below
// Keep this at -1; if height is inverted, flip pitch sign below instead.
static const float COORDS_Z_SIGN = -1.0f;
// If "object higher -> arm goes lower", the camera pitch angle sign is likely inverted.
// Flip this to -1 to invert pitchAngleRad.
static const float COORDS_PITCH_SIGN = -1.0f;

// Single continuous approach from search pose directly to grab target.
static const uint16_t GRAB_APPROACH_STEPS = 200;
static const uint16_t GRAB_APPROACH_MS = 12;
// How far past tY the gripper reaches (mm)
static const float GRAB_OVERSHOOT_MM = 70.0f;
// Extra forward push during plunge so the object sits deeper in the jaws.
static const float GRAB_PLUNGE_FORWARD_OFFSET_MM = 30.0f;
// If GRAB reaches too low under the object, increase GRAB_Z_BIAS_MM.
// GRAB_MIN_Z_MM prevents diving below a safe height even if vision Z is wrong.
static const float GRAB_Z_BIAS_MM = 0.0f;
// Table safety: never allow the grab path to go below this Z (mm).
// Increase if you still see table contact; decrease if it stops too high to grab short objects.
// Floor: raise if arm still dives too low / hits table; lower only if it stops short on tall objects.
static const float GRAB_MIN_Z_MM = 55.0f;
static const float GRAB_MAX_Z_MM = 310.0f;
// Hard ceiling to prevent sudden "snap too high" on GRAB receipt.
static const float GRAB_HARD_MAX_Z_MM = 150.0f;

// Gripper calibration (µs): increase GRIPPER_CLOSE_US if it doesn't clamp tight enough.
static const int GRIPPER_OPEN_US = 900;
// Slightly tighter close for a firmer grab.
static const int GRIPPER_CLOSE_US = 1600;
static const uint16_t GRIPPER_HOLD_MS = 1400;

// Drop-off pose (post-grab): the base rotates this many µs AWAY FROM baseHome
// before opening the gripper. ~11.11 µs per degree -> +1000 µs ≈ +90° rotation
// to the right (clockwise viewed from above). Flip the sign for a left-side drop.
static const int DROP_BASE_OFFSET_US = 1000;

// WAVE gesture: how many µs to raise (extend) the elbow above its wave-entry
// position. Positive = more extended / forearm lifted. If during the wave the
// arm STILL aims down, increase this; if it over-extends, reduce or flip sign.
static const int WAVE_ELB_RAISE_US = 300;

// DEMO gesture ("show us your movement") — gentle full-body wiggle.
// Each joint gets its own amplitude (µs) around its HOME position and a
// distinct phase offset so the whole arm looks alive rather than stiff.
// Keep these moderate: the goal is personality, not acrobatics.
static const int   DEMO_BASE_AMP_US        = 250;   // ~±22°
static const int   DEMO_SHOULDER_AMP_US    = 100;   // small bob (weight-bearing)
static const int   DEMO_ELBOW_AMP_US       = 140;
static const int   DEMO_WRIST_YAW_AMP_US   = 220;
static const int   DEMO_WRIST_PITCH_AMP_US = 130;
static const int   DEMO_WRIST_ROLL_AMP_US  = 350;
static const uint16_t DEMO_CYCLES          = 3;
static const uint16_t DEMO_STEPS_PER_CYC   = 90;
static const uint16_t DEMO_STEP_MS         = 12;
// During CLAMP we should not change Z. Use a small forward nudge instead.
static const float GRAB_PRE_CLAMP_LIFT_MM = 0.0f;
// Extra forward travel right before clamp so the object sits deeper in the jaws.
static const float GRAB_PRE_CLAMP_FORWARD_NUDGE_MM = 18.0f;
// Re-open after a grab so the next target isn't occluded by the gripper.
static const bool OPEN_GRIPPER_AFTER_GRAB = true;
// IK: grip_angle_d is absolute tool angle in the arm Y–Z plane (deg). 0° = tool extends along +Y
// (horizontal in that plane ≈ jaws parallel to the table when the arm is vertical). Trim if slightly tilted.
static const float GRAB_GRIP_ANGLE_DEG = 0.0f;

// Pin Mapping
uint8_t baseMotor = 0;       
uint8_t shoulderMotor = 1;   
uint8_t elbowMotor = 2;      
uint8_t wristYawMotor = 3;   
uint8_t wristPitchMotor = 4; 
uint8_t wristRollMotor = 5;  
uint8_t gripperMotor = 15;    

// State Variables
int currentBasePulse;
int currentPitchPulse;
int baseHome;
int pitchHome;

// During GRAB we can lock out TRACK on specific axes to prevent eye-in-hand
// parallax from rotating the base during forward plunge.
bool g_grab_lock_base = false;
bool g_grab_lock_pitch = false;

// Last commanded arm pulses (for smooth GRAB entry — search pose uses raw PWM, not set_arm)
int g_lastBaseUs;
int g_lastShoulderUs;
int g_lastElbowUs;
int g_lastWristPitchUs;
int g_searchShoulderUs;
int g_searchElbowUs;

// --- SWEEP TRACKING VARIABLES ---
unsigned long lastCommandTime = 0;
int sweepRange = 450;   
int pitchDepth = 200;   
int sweepOffset = -450; 
int sweepStep = 3;      
bool g_sweep_enabled = false;   // disabled until a GRAB starts
// When true, the auto-sweep biases wrist pitch UP (looking for a face) instead of
// doing the downward nod used to look for objects on a table.
bool g_face_search_mode = false;
// How far UP (in µs, subtracted from pitchHome) the pitch tilts during face sweep.
int pitchUpDepth = 280;

// --- DECOUPLED NODDING VARIABLES ---
float nodAngle = 0.0;
float nodSpeed = 0.022; 

// Forward declaration (defined at bottom)
String getValue(String data, char separator, int index);

// Re-write the last known arm pulses (helps if power/noise causes brief drop).
void refresh_arm_hold_pose() {
  pwm.writeMicroseconds(baseMotor, g_lastBaseUs);
  pwm.writeMicroseconds(shoulderMotor, g_lastShoulderUs);
  pwm.writeMicroseconds(elbowMotor, g_lastElbowUs);
  pwm.writeMicroseconds(wristPitchMotor, g_lastWristPitchUs);
}

// Process one serial command during grab.
// Returns: 0 = nothing/TRACK/GET_COORDS handled, 1 = UPDATE received, 2 = CLAMP received, 3 = ABORT received.
int grab_check_serial(float &tX, float &tY, float &tZ) {
  if (!Serial.available()) return 0;
  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  if (cmd.startsWith("UPDATE ")) {
    String data = cmd.substring(7);
    float nx = getValue(data, ',', 0).toFloat();
    float ny = getValue(data, ',', 1).toFloat();
    float nz = getValue(data, ',', 2).toFloat();
    nz += GRAB_Z_BIAS_MM;
    if (nz < GRAB_MIN_Z_MM) nz = GRAB_MIN_Z_MM;
    if (nz > GRAB_MAX_Z_MM) nz = GRAB_MAX_Z_MM;
    tX = nx; tY = ny; tZ = nz;
    return 1;
  }
  if (cmd.startsWith("TRACK ")) {
    int sp = cmd.indexOf(' ');
    String data = cmd.substring(sp + 1);
    int bAdj = getValue(data, ',', 0).toInt();
    int pAdj = getValue(data, ',', 1).toInt();
    if (!g_grab_lock_base) {
      currentBasePulse = constrain(currentBasePulse + bAdj, 600, 2400);
    }
    if (!g_grab_lock_pitch) {
      currentPitchPulse = constrain(currentPitchPulse + pAdj, 600, 2400);
    }
    return 0;
  }
  if (cmd.startsWith("LOCK_BASE ")) {
    int sp = cmd.indexOf(' ');
    int v = cmd.substring(sp + 1).toInt();
    g_grab_lock_base = (v != 0);
    return 0;
  }
  if (cmd.startsWith("LOCK_PITCH ")) {
    int sp = cmd.indexOf(' ');
    int v = cmd.substring(sp + 1).toInt();
    g_grab_lock_pitch = (v != 0);
    return 0;
  }
  if (cmd.startsWith("GET_COORDS ")) {
    int sp = cmd.indexOf(' ');
    float distance = cmd.substring(sp + 1).toFloat();
    calculate_real_coords(distance);
    return 0;
  }
  if (cmd.startsWith("CLAMP")) {
    return 2;
  }
  if (cmd.startsWith("ABORT")) {
    return 3;
  }
  return 0;
}

void go_home_pose_smooth(uint16_t steps, uint16_t msPerStep) {
  // Base 0° (baseHome), other joints to calibrated home.
  float sb0 = (float)g_lastBaseUs;
  float ss0 = (float)g_lastShoulderUs;
  float se0 = (float)g_lastElbowUs;
  float sw0 = (float)g_lastWristPitchUs;

  const float tb = (float)baseHome;
  const float ts = (float)(BASE_CENTER + shoulderOffset);
  const float te = (float)(BASE_CENTER + elbowOffset);
  const float tw = (float)pitchHome;
  const int ty = BASE_CENTER + wristYawOffset;
  const int tr = BASE_CENTER + wristRollOffset;

  if (steps < 1) steps = 1;
  for (uint16_t i = 1; i <= steps; i++) {
    float lin = (float)i / (float)steps;
    float t = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
    pwm.writeMicroseconds(baseMotor, (int)(sb0 + t * (tb - sb0)));
    pwm.writeMicroseconds(shoulderMotor, (int)(ss0 + t * (ts - ss0)));
    pwm.writeMicroseconds(elbowMotor, (int)(se0 + t * (te - se0)));
    pwm.writeMicroseconds(wristPitchMotor, (int)(sw0 + t * (tw - sw0)));
    pwm.writeMicroseconds(wristYawMotor, ty);
    pwm.writeMicroseconds(wristRollMotor, tr);
    delay(msPerStep);
  }

  g_lastBaseUs = (int)tb;
  g_lastShoulderUs = (int)ts;
  g_lastElbowUs = (int)te;
  g_lastWristPitchUs = (int)tw;
  currentBasePulse = g_lastBaseUs;
  currentPitchPulse = g_lastWristPitchUs;
  sweepOffset = 0;
}

void go_search_pose_smooth(uint16_t steps, uint16_t msPerStep) {
  // Search pose: shoulder/elbow to stored search, wrist pitch home, base to left sweep start.
  float sb0 = (float)g_lastBaseUs;
  float ss0 = (float)g_lastShoulderUs;
  float se0 = (float)g_lastElbowUs;
  float sw0 = (float)g_lastWristPitchUs;

  // Start sweep from left edge for consistent behavior.
  sweepOffset = -sweepRange;
  const float tb = (float)(baseHome + sweepOffset);
  const float ts = (float)g_searchShoulderUs;
  const float te = (float)g_searchElbowUs;
  const float tw = (float)pitchHome;

  if (steps < 1) steps = 1;
  for (uint16_t i = 1; i <= steps; i++) {
    float lin = (float)i / (float)steps;
    float t = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
    pwm.writeMicroseconds(baseMotor, (int)(sb0 + t * (tb - sb0)));
    pwm.writeMicroseconds(shoulderMotor, (int)(ss0 + t * (ts - ss0)));
    pwm.writeMicroseconds(elbowMotor, (int)(se0 + t * (te - se0)));
    pwm.writeMicroseconds(wristPitchMotor, (int)(sw0 + t * (tw - sw0)));
    delay(msPerStep);
  }

  g_lastBaseUs = (int)tb;
  g_lastShoulderUs = (int)ts;
  g_lastElbowUs = (int)te;
  g_lastWristPitchUs = (int)tw;
  currentBasePulse = g_lastBaseUs;
  currentPitchPulse = g_lastWristPitchUs;
}

void setup() {
  Serial.begin(115200); 
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(50);
  
  baseHome = BASE_CENTER + baseOffset;
  pitchHome = BASE_CENTER + wristPitchOffset;

  // Startup settle
  // Disable outputs during settle to avoid snapping from an unknown power-off pose.
  for (int ch = 0; ch < 16; ch++) {
    // PCA9685 "full OFF" uses off=4096 (special bit).
    pwm.setPWM(ch, 0, 4096);
  }
  delay(5000);

  // 1. Move to Calibrated Home (smooth)
  Serial.println("Moving to Home...");
  // Smooth the startup motion to reduce snapping.
  int shlHome = BASE_CENTER + shoulderOffset;
  int elbHome = BASE_CENTER + elbowOffset;
  int yawHome = BASE_CENTER + wristYawOffset;
  int rolHome = BASE_CENTER + wristRollOffset;
  pwm.writeMicroseconds(gripperMotor, 900);
  for (int i = 0; i <= 80; i++) {
    float t = (float)i / 80.0f;
    float s = (1.0f - cosf(t * 3.14159265f)) * 0.5f;
    pwm.writeMicroseconds(baseMotor, (int)(BASE_CENTER + s * (baseHome - BASE_CENTER)));
    pwm.writeMicroseconds(shoulderMotor, (int)(BASE_CENTER + s * (shlHome - BASE_CENTER)));
    pwm.writeMicroseconds(elbowMotor, (int)(BASE_CENTER + s * (elbHome - BASE_CENTER)));
    pwm.writeMicroseconds(wristYawMotor, (int)(BASE_CENTER + s * (yawHome - BASE_CENTER)));
    pwm.writeMicroseconds(wristPitchMotor, (int)(BASE_CENTER + s * (pitchHome - BASE_CENTER)));
    pwm.writeMicroseconds(wristRollMotor, (int)(BASE_CENTER + s * (rolHome - BASE_CENTER)));
    delay(18);
  }
  delay(350);
  
  pwm.setPWM(gripperMotor, 0, 0);
  delay(300);

  // Store search pose (do not move there until GRAB)
  g_searchShoulderUs = (BASE_CENTER + shoulderOffset) - 388;
  g_searchElbowUs = (BASE_CENTER + elbowOffset) - 600;

  // Initialize "hold at home" positions
  currentBasePulse = baseHome;
  currentPitchPulse = pitchHome;
  g_lastBaseUs = baseHome;
  g_lastShoulderUs = BASE_CENTER + shoulderOffset;
  g_lastElbowUs = BASE_CENTER + elbowOffset;
  g_lastWristPitchUs = pitchHome;

  g_sweep_enabled = false;
  sweepOffset = 0;

  Serial.println("READY");
  Serial.println("FW_BUILD_V9_NOD_SHAKE_POINT_2026");
}

// IK only — same math as before; outputs constrained µs (set_arm body unchanged, refactored).
void compute_arm_us(float x, float y, float z, float grip_angle_d,
                    float* outBas, float* outShl, float* outElb, float* outWri) {
  float bas_angle_r = atan2(x, y);
  float rdist = sqrt(x * x + y * y);
  y = rdist;

  float grip_angle_r = radians(grip_angle_d);
  float grip_off_z = sin(grip_angle_r) * GRIPPER;
  float grip_off_y = cos(grip_angle_r) * GRIPPER;

  float wrist_z = (z - grip_off_z) - BASE_HGT;
  float wrist_y = y - grip_off_y;

  float s_w_sq = (wrist_z * wrist_z) + (wrist_y * wrist_y);
  float s_w = sqrt(s_w_sq);

  if (s_w > (HUMERUS + ULNA)) {
    s_w = (HUMERUS + ULNA) - 0.1f;
    s_w_sq = s_w * s_w;
  }

  float a1 = atan2(wrist_z, wrist_y);
  float cos_a2 = (hum_sq - uln_sq + s_w_sq) / (2.0f * HUMERUS * s_w);
  float a2 = acos(constrain(cos_a2, -1.0, 1.0));
  float shl_angle_d = degrees(a1 + a2);

  float cos_elb = (hum_sq + uln_sq - s_w_sq) / (2.0f * HUMERUS * ULNA);
  float elb_angle_d = degrees(acos(constrain(cos_elb, -1.0, 1.0)));

  float absolute_ulna_angle = shl_angle_d + elb_angle_d - 180.0f;
  float relative_wrist_angle = grip_angle_d - absolute_ulna_angle;

  *outBas = (1500.0f + baseOffset) - (degrees(bas_angle_r) * 11.11f);
  *outShl = (1500.0f + shoulderOffset) - ((shl_angle_d - 90.0f) * 11.11f);
  *outElb = (1500.0f + elbowOffset) + ((elb_angle_d - 90.0f) * 11.11f);
  *outWri = (1500.0f + wristPitchOffset) - (relative_wrist_angle * 7.41f);

  *outBas = constrain(*outBas, 600.0f, 2400.0f);
  *outShl = constrain(*outShl, 600.0f, 2400.0f);
  *outElb = constrain(*outElb, 600.0f, 2400.0f);
  *outWri = constrain(*outWri, 600.0f, 2400.0f);
}

// Ease-in-out blend in joint µs (smoother than linear — less "snap" at start/end)
void move_arm_smooth_to_xyz(float x, float y, float z, float grip_angle_d, uint16_t steps, uint16_t msPerStep) {
  float tb, ts, te, tw;
  compute_arm_us(x, y, z, grip_angle_d, &tb, &ts, &te, &tw);
  float sb = (float)g_lastBaseUs;
  float ss = (float)g_lastShoulderUs;
  float se = (float)g_lastElbowUs;
  float sw = (float)g_lastWristPitchUs;
  if (steps < 1) steps = 1;
  for (uint16_t i = 1; i <= steps; i++) {
    float lin = (float)i / (float)steps;
    float t = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
    pwm.writeMicroseconds(baseMotor, (int)(sb + t * (tb - sb)));
    pwm.writeMicroseconds(shoulderMotor, (int)(ss + t * (ts - ss)));
    pwm.writeMicroseconds(elbowMotor, (int)(se + t * (te - se)));
    pwm.writeMicroseconds(wristPitchMotor, (int)(sw + t * (tw - sw)));
    delay(msPerStep);
  }
  g_lastBaseUs = (int)tb;
  g_lastShoulderUs = (int)ts;
  g_lastElbowUs = (int)te;
  g_lastWristPitchUs = (int)tw;
}

void return_to_search_pose(uint16_t steps, uint16_t msPerStep) {
  float ss0 = (float)g_lastShoulderUs;
  float se0 = (float)g_lastElbowUs;
  float sw0 = (float)g_lastWristPitchUs;
  float sb0 = (float)g_lastBaseUs;

  float ssT = (float)g_searchShoulderUs;
  float seT = (float)g_searchElbowUs;
  float swT = (float)pitchHome;

  if (steps < 1) {
    steps = 1;
  }
  for (uint16_t i = 1; i <= steps; i++) {
    float lin = (float)i / (float)steps;
    float t = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
    pwm.writeMicroseconds(baseMotor, (int)sb0); // keep base steady
    pwm.writeMicroseconds(shoulderMotor, (int)(ss0 + t * (ssT - ss0)));
    pwm.writeMicroseconds(elbowMotor, (int)(se0 + t * (seT - se0)));
    pwm.writeMicroseconds(wristPitchMotor, (int)(sw0 + t * (swT - sw0)));
    delay(msPerStep);
  }

  g_lastShoulderUs = (int)ssT;
  g_lastElbowUs = (int)seT;
  g_lastWristPitchUs = (int)swT;

  // Ensure TRACK/autosweep starts from the real commanded pulses
  currentBasePulse = g_lastBaseUs;
  currentPitchPulse = g_lastWristPitchUs;

  if (OPEN_GRIPPER_AFTER_GRAB) {
    for (int p = GRIPPER_CLOSE_US; p >= GRIPPER_OPEN_US; p -= 12) {
      pwm.writeMicroseconds(gripperMotor, p);
      delay(12);
    }
  }
}

void return_to_home_drop_pose(uint16_t steps, uint16_t msPerStep) {
  // DROP pose: base rotated to baseHome + DROP_BASE_OFFSET_US (~90° from forward)
  // so the object is released off to the side instead of right in front of the camera.
  // Shoulder/elbow/wrist-pitch all move to their calibrated home values.
  // Gripper stays CLOSED during the move; we open after arriving to drop.
  float sb0 = (float)g_lastBaseUs;
  float ss0 = (float)g_lastShoulderUs;
  float se0 = (float)g_lastElbowUs;
  float sw0 = (float)g_lastWristPitchUs;

  // Drop base target, constrained to the safe servo range.
  int dropBase = baseHome + DROP_BASE_OFFSET_US;
  if (dropBase < 600)  dropBase = 600;
  if (dropBase > 2400) dropBase = 2400;
  const float tb = (float)dropBase;
  const float ts = (float)(BASE_CENTER + shoulderOffset);
  const float te = (float)(BASE_CENTER + elbowOffset);
  const float tw = (float)pitchHome;
  const int ty = BASE_CENTER + wristYawOffset;
  const int tr = BASE_CENTER + wristRollOffset;

  if (steps < 1) {
    steps = 1;
  }

  for (uint16_t i = 1; i <= steps; i++) {
    float lin = (float)i / (float)steps;
    float t = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
    float cb = sb0 + t * (tb - sb0);
    float cs = ss0 + t * (ts - ss0);
    float ce = se0 + t * (te - se0);
    float cw = sw0 + t * (tw - sw0);

    pwm.writeMicroseconds(baseMotor, (int)cb);
    pwm.writeMicroseconds(shoulderMotor, (int)cs);
    pwm.writeMicroseconds(elbowMotor, (int)ce);
    pwm.writeMicroseconds(wristPitchMotor, (int)cw);
    pwm.writeMicroseconds(wristYawMotor, ty);
    pwm.writeMicroseconds(wristRollMotor, tr);

    delay(msPerStep);
  }

  g_lastBaseUs = (int)tb;
  g_lastShoulderUs = (int)ts;
  g_lastElbowUs = (int)te;
  g_lastWristPitchUs = (int)tw;

  // Ensure TRACK/autosweep starts from the real commanded pulses
  currentBasePulse = g_lastBaseUs;
  sweepOffset = 0;
  currentPitchPulse = g_lastWristPitchUs;

  // Drop: open gripper
  for (int p = GRIPPER_CLOSE_US; p >= GRIPPER_OPEN_US; p -= 12) {
    pwm.writeMicroseconds(gripperMotor, p);
    delay(12);
  }
}

void set_arm(float x, float y, float z, float grip_angle_d, int servoSpeed) {
  float bas_servopulse, shl_servopulse, elb_servopulse, wri_servopulse;
  compute_arm_us(x, y, z, grip_angle_d, &bas_servopulse, &shl_servopulse, &elb_servopulse, &wri_servopulse);

  pwm.writeMicroseconds(baseMotor, (int)bas_servopulse);
  pwm.writeMicroseconds(shoulderMotor, (int)shl_servopulse);
  pwm.writeMicroseconds(elbowMotor, (int)elb_servopulse);
  pwm.writeMicroseconds(wristPitchMotor, (int)wri_servopulse);

  g_lastBaseUs = (int)bas_servopulse;
  g_lastShoulderUs = (int)shl_servopulse;
  g_lastElbowUs = (int)elb_servopulse;
  g_lastWristPitchUs = (int)wri_servopulse;
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    // START_FACE_SEARCH: same as START_SEARCH but biases the sweep to look UP
    // (looking for a human face) instead of nodding DOWN to look for objects.
    // Must be checked BEFORE START_SEARCH since startsWith("START_SEARCH") is a prefix match.
    if (cmd.startsWith("START_FACE_SEARCH")) {
      g_face_search_mode = true;
      if (!g_sweep_enabled) {
        go_search_pose_smooth(60, 11);
        g_sweep_enabled = true;
      }
      lastCommandTime = millis();
      Serial.println("SEARCH_READY");
    }
    else if (cmd.startsWith("START_SEARCH")) {
      g_face_search_mode = false;
      if (!g_sweep_enabled) {
        go_search_pose_smooth(60, 11);
        g_sweep_enabled = true;
      }
      lastCommandTime = millis();
      Serial.println("SEARCH_READY");
    }

    else if (cmd.startsWith("TRACK")) {
      lastCommandTime = millis(); 

      // If the vision loop starts sending TRACK before any GRAB, enter the sweep/search
      // pose now (this is the "go into sweep when detection starts" behavior).
      // Faster transition (~660ms) to reduce the perceived delay after a voice command.
      if (!g_sweep_enabled) {
        g_sweep_enabled = true;
        go_search_pose_smooth(60, 11);
        lastCommandTime = millis();
      }
      
      int firstSpace = cmd.indexOf(' ');
      String data = cmd.substring(firstSpace + 1);
      int bAdj = getValue(data, ',', 0).toInt();
      int pAdj = getValue(data, ',', 1).toInt();

      currentBasePulse = constrain(currentBasePulse + bAdj, 600, 2400);
      currentPitchPulse = constrain(currentPitchPulse + pAdj, 600, 2400);

      pwm.writeMicroseconds(baseMotor, currentBasePulse);
      pwm.writeMicroseconds(wristPitchMotor, currentPitchPulse);

      g_lastBaseUs = currentBasePulse;
      g_lastWristPitchUs = currentPitchPulse;

      sweepOffset = constrain(currentBasePulse - baseHome, -sweepRange, sweepRange);
    }
    
    else if (cmd.startsWith("GET_COORDS")) {
      // Freeze auto-sweep while we compute COORDS so pitch/base don't change mid-measurement.
      lastCommandTime = millis();
      int firstSpace = cmd.indexOf(' ');
      float distance = cmd.substring(firstSpace + 1).toFloat();
      calculate_real_coords(distance);
    }
    else if (cmd.startsWith("WAVE")) {
      // "WAVE" gesture:
      //   - Simultaneously ease wrist pitch back to 90° AND raise the elbow by
      //     WAVE_ELB_RAISE_US so the arm isn't aimed down while waving.
      //   - Close the gripper (a "closed hand" waving).
      //   - Wiggle wrist ROLL side-to-side with a subtle ELBOW bob around the
      //     RAISED elbow position (not the original low one).
      //   - Smooth-return to HOME and re-open the gripper.
      lastCommandTime = millis();

      const int rollCenter = BASE_CENTER + wristRollOffset;
      const int startElb   = g_lastElbowUs;
      int raisedElb = startElb + WAVE_ELB_RAISE_US;
      if (raisedElb < 600)  raisedElb = 600;
      if (raisedElb > 2400) raisedElb = 2400;
      int lockedBase  = currentBasePulse;
      int startPitch  = currentPitchPulse;
      int lockedPitch = pitchHome;   // target pitch during the wave

      Serial.println("WAVE_START");

      // 1. Simultaneously: ease pitch -> 90° AND ease elbow -> raisedElb.
      {
        const uint16_t preSteps = 48;
        for (uint16_t i = 1; i <= preSteps; i++) {
          float lin = (float)i / (float)preSteps;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          int pp = (int)((float)startPitch + s * (float)(lockedPitch - startPitch));
          int ee = (int)((float)startElb   + s * (float)(raisedElb   - startElb));
          pwm.writeMicroseconds(wristPitchMotor, pp);
          pwm.writeMicroseconds(elbowMotor,      ee);
          pwm.writeMicroseconds(baseMotor, lockedBase);
          pwm.writeMicroseconds(wristRollMotor, rollCenter);
          delay(12);
        }
        currentPitchPulse  = lockedPitch;
        g_lastWristPitchUs = lockedPitch;
        g_lastElbowUs      = raisedElb;
      }

      // 2. Close gripper ("closed hand" wave).
      for (int p = GRIPPER_OPEN_US; p <= GRIPPER_CLOSE_US; p += 20) {
        pwm.writeMicroseconds(gripperMotor, p);
        delay(8);
      }

      // 3. Wiggle wrist roll (main wave) + elbow bob AROUND the raised elbow.
      const uint16_t cycles       = 3;
      const uint16_t stepsPerCyc  = 70;
      const float    rollAmp      = 400.0f;  // wrist roll swing (us)
      const float    elbAmp       = 80.0f;   // elbow bob (us) around raisedElb

      bool wave_aborted = false;
      for (uint16_t c = 0; c < cycles && !wave_aborted; c++) {
        for (uint16_t i = 0; i < stepsPerCyc; i++) {
          if (Serial.available()) {
            String sub = Serial.readStringUntil('\n');
            sub.trim();
            if (sub.startsWith("ABORT")) { wave_aborted = true; break; }
          }
          float phase  = (float)i / (float)stepsPerCyc;
          float angle  = phase * 2.0f * 3.14159265f;
          int   rollUs = rollCenter + (int)(rollAmp * sinf(angle));
          int   elbUs  = raisedElb  + (int)(elbAmp  * sinf(angle + 1.5707963f)); // 90° offset
          pwm.writeMicroseconds(wristRollMotor, rollUs);
          pwm.writeMicroseconds(elbowMotor,     elbUs);
          pwm.writeMicroseconds(baseMotor,       lockedBase);
          pwm.writeMicroseconds(wristPitchMotor, lockedPitch);
          delay(12);
        }
      }

      // 4. Settle wrist roll + elbow back to neutral (hold elbow at raisedElb).
      {
        const uint16_t settle = 40;
        for (uint16_t i = 1; i <= settle; i++) {
          pwm.writeMicroseconds(wristRollMotor, rollCenter);
          pwm.writeMicroseconds(elbowMotor,     raisedElb);
          pwm.writeMicroseconds(baseMotor,       lockedBase);
          pwm.writeMicroseconds(wristPitchMotor, lockedPitch);
          delay(14);
        }
      }
      g_lastElbowUs      = raisedElb;
      g_lastBaseUs       = lockedBase;
      g_lastWristPitchUs = lockedPitch;

      // 5. Smoothly return to HOME (go_home_pose_smooth interpolates elbow from
      //    raisedElb back to its calibrated home).
      go_home_pose_smooth(90, 14);

      // 6. Re-open the gripper once we're back at HOME.
      for (int p = GRIPPER_CLOSE_US; p >= GRIPPER_OPEN_US; p -= 12) {
        pwm.writeMicroseconds(gripperMotor, p);
        delay(12);
      }

      g_sweep_enabled = false;
      g_face_search_mode = false;

      Serial.println("WAVE_COMPLETE");
      lastCommandTime = millis();
    }

    else if (cmd.startsWith("DEMO")) {
      // "DEMO" gesture ("show us your movement"):
      //   Gentle full-body wiggle of ALL six joints around their calibrated
      //   HOME positions, with per-joint phase offsets so the motion looks
      //   organic. Finishes with a smooth return to HOME. ABORT-safe.
      lastCommandTime = millis();

      const int baseC     = baseHome;
      const int shoulderC = BASE_CENTER + shoulderOffset;
      const int elbowC    = BASE_CENTER + elbowOffset;
      const int yawC      = BASE_CENTER + wristYawOffset;
      const int pitchC    = pitchHome;
      const int rollC     = BASE_CENTER + wristRollOffset;

      // 1. Ease everything from wherever we are now to HOME so the wiggle
      //    starts from a known reference — same logic as go_home_pose_smooth
      //    but we stop BEFORE the final servo writes so we can hand off to
      //    the sinusoidal loop without a visible pause.
      {
        float sb0 = (float)g_lastBaseUs;
        float ss0 = (float)g_lastShoulderUs;
        float se0 = (float)g_lastElbowUs;
        float sw0 = (float)g_lastWristPitchUs;
        const uint16_t preSteps = 50;
        for (uint16_t i = 1; i <= preSteps; i++) {
          float lin = (float)i / (float)preSteps;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          pwm.writeMicroseconds(baseMotor,       (int)(sb0 + s * (baseC     - sb0)));
          pwm.writeMicroseconds(shoulderMotor,   (int)(ss0 + s * (shoulderC - ss0)));
          pwm.writeMicroseconds(elbowMotor,      (int)(se0 + s * (elbowC    - se0)));
          pwm.writeMicroseconds(wristPitchMotor, (int)(sw0 + s * (pitchC    - sw0)));
          pwm.writeMicroseconds(wristYawMotor,   yawC);
          pwm.writeMicroseconds(wristRollMotor,  rollC);
          delay(10);
        }
        g_lastBaseUs       = baseC;
        g_lastShoulderUs   = shoulderC;
        g_lastElbowUs      = elbowC;
        g_lastWristPitchUs = pitchC;
        currentBasePulse   = baseC;
        currentPitchPulse  = pitchC;
      }

      Serial.println("DEMO_START");

      // 2. Multi-joint sinusoidal wiggle with staggered phases so the arm
      //    looks expressive, not mechanical.
      const float TAU = 2.0f * 3.14159265f;
      const float phBase  = 0.0f;
      const float phShoul = TAU / 6.0f;
      const float phElbow = 2.0f * TAU / 6.0f;
      const float phYaw   = 3.0f * TAU / 6.0f;
      const float phPitch = 4.0f * TAU / 6.0f;
      const float phRoll  = 5.0f * TAU / 6.0f;

      bool demo_aborted = false;
      for (uint16_t c = 0; c < DEMO_CYCLES && !demo_aborted; c++) {
        for (uint16_t i = 0; i < DEMO_STEPS_PER_CYC; i++) {
          if (Serial.available()) {
            String sub = Serial.readStringUntil('\n');
            sub.trim();
            if (sub.startsWith("ABORT")) { demo_aborted = true; break; }
          }
          float phase = ((float)i / (float)DEMO_STEPS_PER_CYC) * TAU;

          // Envelope: ramp amplitude in at the start and out at the end so
          // the motion begins and ends gracefully instead of snapping.
          float envT = ((float)c + (float)i / (float)DEMO_STEPS_PER_CYC)
                       / (float)DEMO_CYCLES;  // 0..1 across whole demo
          float env  = sinf(envT * 3.14159265f); // 0 at edges, 1 in middle
          if (env < 0.0f) env = 0.0f;
          if (env > 1.0f) env = 1.0f;
          // Keep a baseline so the middle of the demo still looks lively.
          float amp = 0.35f + 0.65f * env;

          int baseUs  = baseC     + (int)(amp * (float)DEMO_BASE_AMP_US        * sinf(phase + phBase));
          int shouUs  = shoulderC + (int)(amp * (float)DEMO_SHOULDER_AMP_US    * sinf(phase + phShoul));
          int elbUs   = elbowC    + (int)(amp * (float)DEMO_ELBOW_AMP_US       * sinf(phase + phElbow));
          int yawUs   = yawC      + (int)(amp * (float)DEMO_WRIST_YAW_AMP_US   * sinf(phase + phYaw));
          int pitUs   = pitchC    + (int)(amp * (float)DEMO_WRIST_PITCH_AMP_US * sinf(phase + phPitch));
          int rolUs   = rollC     + (int)(amp * (float)DEMO_WRIST_ROLL_AMP_US  * sinf(phase + phRoll));

          // Safety clamp so no servo is ever driven outside the usable range.
          if (baseUs < 600) baseUs = 600; if (baseUs > 2400) baseUs = 2400;
          if (shouUs < 600) shouUs = 600; if (shouUs > 2400) shouUs = 2400;
          if (elbUs  < 600) elbUs  = 600; if (elbUs  > 2400) elbUs  = 2400;
          if (yawUs  < 600) yawUs  = 600; if (yawUs  > 2400) yawUs  = 2400;
          if (pitUs  < 600) pitUs  = 600; if (pitUs  > 2400) pitUs  = 2400;
          if (rolUs  < 600) rolUs  = 600; if (rolUs  > 2400) rolUs  = 2400;

          pwm.writeMicroseconds(baseMotor,       baseUs);
          pwm.writeMicroseconds(shoulderMotor,   shouUs);
          pwm.writeMicroseconds(elbowMotor,      elbUs);
          pwm.writeMicroseconds(wristYawMotor,   yawUs);
          pwm.writeMicroseconds(wristPitchMotor, pitUs);
          pwm.writeMicroseconds(wristRollMotor,  rolUs);
          delay(DEMO_STEP_MS);
        }
      }

      // 3. Always leave the arm at a clean HOME pose — even on abort —
      //    so subsequent commands start from a known state.
      g_lastBaseUs       = baseC;
      g_lastShoulderUs   = shoulderC;
      g_lastElbowUs      = elbowC;
      g_lastWristPitchUs = pitchC;
      currentBasePulse   = baseC;
      currentPitchPulse  = pitchC;
      go_home_pose_smooth(60, 12);

      g_sweep_enabled = false;
      g_face_search_mode = false;

      Serial.println(demo_aborted ? "DEMO_ABORTED" : "DEMO_COMPLETE");
      lastCommandTime = millis();
    }

    else if (cmd.startsWith("NOD")) {
      // "NOD" gesture — a little "yes" head-bob.
      // Oscillates wrist PITCH up/down around pitchHome with small amplitude
      // so it reads clearly as a nod without the arm lurching. ABORT-safe.
      lastCommandTime = millis();

      const int pitchCenter = pitchHome;
      const int startPitch  = currentPitchPulse;

      // 1. Ease wrist pitch back to pitchHome so the nod starts from a
      //    consistent reference regardless of prior pose.
      {
        const uint16_t pre = 30;
        for (uint16_t i = 1; i <= pre; i++) {
          float lin = (float)i / (float)pre;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          int pp = (int)((float)startPitch + s * (float)(pitchCenter - startPitch));
          pwm.writeMicroseconds(wristPitchMotor, pp);
          delay(10);
        }
        currentPitchPulse  = pitchCenter;
        g_lastWristPitchUs = pitchCenter;
      }

      Serial.println("NOD_START");

      // 2. Two clear nod cycles: pitch sweeps DOWN then UP around center.
      const int      nodAmp      = 180;   // ~±24°
      const uint16_t nodCycles   = 2;
      const uint16_t nodSteps    = 48;
      const uint16_t nodStepMs   = 10;
      bool nod_aborted = false;
      for (uint16_t c = 0; c < nodCycles && !nod_aborted; c++) {
        for (uint16_t i = 0; i < nodSteps; i++) {
          if (Serial.available()) {
            String sub = Serial.readStringUntil('\n');
            sub.trim();
            if (sub.startsWith("ABORT")) { nod_aborted = true; break; }
          }
          float phase = ((float)i / (float)nodSteps) * 2.0f * 3.14159265f;
          int pitUs = pitchCenter + (int)((float)nodAmp * sinf(phase));
          if (pitUs < 600)  pitUs = 600;
          if (pitUs > 2400) pitUs = 2400;
          pwm.writeMicroseconds(wristPitchMotor, pitUs);
          delay(nodStepMs);
        }
      }

      // 3. Settle back to exact center so subsequent commands have a
      //    clean reference pose.
      pwm.writeMicroseconds(wristPitchMotor, pitchCenter);
      currentPitchPulse  = pitchCenter;
      g_lastWristPitchUs = pitchCenter;

      g_sweep_enabled = false;
      g_face_search_mode = false;
      Serial.println(nod_aborted ? "NOD_ABORTED" : "NOD_COMPLETE");
      lastCommandTime = millis();
    }

    else if (cmd.startsWith("SHAKE")) {
      // "SHAKE" gesture — a little "no" head-shake.
      // Oscillates the BASE left/right around baseHome with a modest
      // amplitude. Reads as the robot literally turning its head.
      lastCommandTime = millis();

      const int baseCenter = baseHome;
      const int startBase  = currentBasePulse;

      // 1. Ease base back to baseHome first.
      {
        const uint16_t pre = 30;
        for (uint16_t i = 1; i <= pre; i++) {
          float lin = (float)i / (float)pre;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          int bb = (int)((float)startBase + s * (float)(baseCenter - startBase));
          pwm.writeMicroseconds(baseMotor, bb);
          delay(10);
        }
        currentBasePulse = baseCenter;
        g_lastBaseUs     = baseCenter;
      }

      Serial.println("SHAKE_START");

      // 2. Two shake cycles. Base is a heavy joint — keep amp moderate.
      const int      shakeAmp    = 160;   // ~±14°
      const uint16_t shakeCycles = 2;
      const uint16_t shakeSteps  = 48;
      const uint16_t shakeStepMs = 10;
      bool shake_aborted = false;
      for (uint16_t c = 0; c < shakeCycles && !shake_aborted; c++) {
        for (uint16_t i = 0; i < shakeSteps; i++) {
          if (Serial.available()) {
            String sub = Serial.readStringUntil('\n');
            sub.trim();
            if (sub.startsWith("ABORT")) { shake_aborted = true; break; }
          }
          float phase = ((float)i / (float)shakeSteps) * 2.0f * 3.14159265f;
          int baseUs = baseCenter + (int)((float)shakeAmp * sinf(phase));
          if (baseUs < 600)  baseUs = 600;
          if (baseUs > 2400) baseUs = 2400;
          pwm.writeMicroseconds(baseMotor, baseUs);
          delay(shakeStepMs);
        }
      }

      // 3. Settle at center.
      pwm.writeMicroseconds(baseMotor, baseCenter);
      currentBasePulse = baseCenter;
      g_lastBaseUs     = baseCenter;

      g_sweep_enabled = false;
      g_face_search_mode = false;
      Serial.println(shake_aborted ? "SHAKE_ABORTED" : "SHAKE_COMPLETE");
      lastCommandTime = millis();
    }

    else if (cmd.startsWith("POINT")) {
      // "POINT x,y,z" — aim at a detected object without grabbing.
      // Reuses the same coordinates Python produces via the TRACK/GET_COORDS
      // loop, but only reaches to a fraction of the way so the gripper
      // clearly indicates the object instead of touching it.
      int firstSpace = cmd.indexOf(' ');
      if (firstSpace <= 0) {
        Serial.println("POINT_ERR no_args");
        return;
      }
      String data = cmd.substring(firstSpace + 1);
      float tX = getValue(data, ',', 0).toFloat();
      float tY = getValue(data, ',', 1).toFloat();
      float tZ = getValue(data, ',', 2).toFloat();

      // Python only sends POINT after the tracking loop has already put the
      // arm into the search pose (same as GRAB). If not, mirror GRAB's
      // entry so the IK math starts from a known config.
      if (!g_sweep_enabled) {
        g_sweep_enabled = true;
        go_search_pose_smooth(60, 11);
      }
      lastCommandTime = millis();

      // Scale the reach so the gripper ends up clearly SHORT of the object.
      // Scaling (x,y) by the same factor preserves atan2(x,y) -> the base
      // angle is unchanged; only the radial distance shrinks.
      const float POINT_REACH_FRAC = 0.62f;
      const float POINT_MIN_Y_MM   = 170.0f;
      const float POINT_Z_LIFT_MM  = 25.0f;   // aim from slightly above, reads more natural
      const float POINT_MIN_Z_MM   = 90.0f;
      const float POINT_MAX_Z_MM   = 320.0f;

      float pX = tX * POINT_REACH_FRAC;
      float pY = tY * POINT_REACH_FRAC;
      float pZ = tZ + POINT_Z_LIFT_MM;

      // If the scaled Y is too close to the base, pull the whole (pX,pY)
      // vector outward so we keep the same direction but meet min reach.
      if (pY < POINT_MIN_Y_MM && pY > 0.1f) {
        float s = POINT_MIN_Y_MM / pY;
        pX *= s;
        pY *= s;
      }
      if (pZ < POINT_MIN_Z_MM) pZ = POINT_MIN_Z_MM;
      if (pZ > POINT_MAX_Z_MM) pZ = POINT_MAX_Z_MM;

      Serial.print("POINT_START target=");
      Serial.print(tX, 1); Serial.print(",");
      Serial.print(tY, 1); Serial.print(",");
      Serial.print(tZ, 1); Serial.print("  stance=");
      Serial.print(pX, 1); Serial.print(",");
      Serial.print(pY, 1); Serial.print(",");
      Serial.println(pZ, 1);

      // 1. Smooth extend into the pointing stance with the gripper held
      //    horizontal (grip_angle_d = 0) so it looks like a clean point.
      move_arm_smooth_to_xyz(pX, pY, pZ, 0.0f, 170, 12);

      // 2. Hold ~1.6s so observers see the point. ABORT-safe.
      unsigned long holdUntil = millis() + 1600UL;
      bool point_aborted = false;
      while (millis() < holdUntil) {
        if (Serial.available()) {
          String sub = Serial.readStringUntil('\n');
          sub.trim();
          if (sub.startsWith("ABORT")) { point_aborted = true; break; }
        }
        delay(15);
      }

      // 3. Smooth return to HOME (base also returns to baseHome).
      go_home_pose_smooth(100, 13);

      g_sweep_enabled = false;
      g_face_search_mode = false;
      Serial.println(point_aborted ? "POINT_ABORTED" : "POINT_COMPLETE");
      lastCommandTime = millis();
    }

    else if (cmd.startsWith("TEST_IK")) {
      int firstSpace = cmd.indexOf(' ');
      String data = cmd.substring(firstSpace + 1);
      
      float tX = getValue(data, ',', 0).toFloat();
      float tY = getValue(data, ',', 1).toFloat();
      float tZ = getValue(data, ',', 2).toFloat();
      int testSpeed = getValue(data, ',', 3).toInt(); 

      set_arm(tX, tY, tZ, 0, testSpeed); 
      delay(5000); 
      lastCommandTime = millis();
    }
    
    // ==========================================
    // GRAB with visual servoing: direct approach from tracked pose,
    // checks serial for UPDATE x,y,z corrections each step.
    // ==========================================
    else if (cmd.startsWith("GRAB")) {
      int firstSpace = cmd.indexOf(' ');
      String data = cmd.substring(firstSpace + 1);

      // Only move into search pose here if we haven't already (TRACK does it first).
      // Skipping this when tracking avoids ~1.3s of wasted motion and preserves the
      // tracked base angle so the arm grabs in the right direction.
      if (!g_sweep_enabled) {
        g_sweep_enabled = true;
        go_search_pose_smooth(60, 11);
      }
      lastCommandTime = millis();

      // --- FINAL GRAB OFFSETS (tune these) ---
      // +38mm over previous 15 = ~1.5 inches farther forward on every grab.
      float FORWARD_OFFSET = 53.0f; // Adds to tY (slight forward push)
      float HEIGHT_OFFSET  = 26.0f;  // Adds to tZ — +12mm vs 14 (less low on object)

      float tX = getValue(data, ',', 0).toFloat();
      float tY = getValue(data, ',', 1).toFloat();
      float tZ = getValue(data, ',', 2).toFloat();

      tY += FORWARD_OFFSET;
      tZ += HEIGHT_OFFSET;

      Serial.print("RECEIVED GRAB Z: ");
      Serial.println(tZ);
      tZ += GRAB_Z_BIAS_MM;
      if (tZ < GRAB_MIN_Z_MM) tZ = GRAB_MIN_Z_MM;
      if (tZ > GRAB_MAX_Z_MM) tZ = GRAB_MAX_Z_MM;
      if (tZ > GRAB_HARD_MAX_Z_MM) tZ = GRAB_HARD_MAX_Z_MM;

      Serial.println("GRAB_START");
      g_grab_lock_base = false;
      g_grab_lock_pitch = false;
      g_lastBaseUs = currentBasePulse;
      g_lastWristPitchUs = currentPitchPulse;

      // STRICT BASE LOCK + SEPARATED AXES:
      // - Lock base servo for the entire GRAB (ignore atan2 changes as Y changes).
      // - Move Z first at fixed Y, then open gripper, then glide Y directly to final tY.
      bool got_clamp = false;
      bool aborted = false;

      // Persist these for later (lift)
      int lockedBaseUs = currentBasePulse;
      float goalY = 120.0f;
      float y_start = 120.0f;
      float current_grab_y = 120.0f; // last Y from write_sew_for — use for lift (avoids jab if CLAMP early)

      // Helper: compute only SEW for x=0, with base locked externally.
      auto write_sew_for = [&](float y, float z) {
        current_grab_y = y;
        float tb, ts, te, tw;
        compute_arm_us(0.0f, y, z, GRAB_GRIP_ANGLE_DEG, &tb, &ts, &te, &tw);
        pwm.writeMicroseconds(baseMotor, lockedBaseUs);
        pwm.writeMicroseconds(shoulderMotor, (int)ts);
        pwm.writeMicroseconds(elbowMotor, (int)te);
        pwm.writeMicroseconds(wristPitchMotor, (int)tw);
        g_lastBaseUs = lockedBaseUs;
        g_lastShoulderUs = (int)ts;
        g_lastElbowUs = (int)te;
        g_lastWristPitchUs = (int)tw;
      };

      {
        // Hard lock base now and never let IK overwrite it.
        g_grab_lock_base = true;
        lockedBaseUs = currentBasePulse;
        pwm.writeMicroseconds(baseMotor, lockedBaseUs);
        g_lastBaseUs = lockedBaseUs;

        // Project world (tX,tY) onto the locked base forward axis so we only move forward.
        float baseAngleDeg = (baseHome - lockedBaseUs) / 11.11f;
        float baseAngleRad = baseAngleDeg * (PI / 180.0f);
        float y_proj = (tX * sinf(baseAngleRad)) + (tY * cosf(baseAngleRad));
        goalY = y_proj; // final distance provided by Python (no extra overshoot)
        if (goalY < 80.0f) goalY = 80.0f;

        // Phase A: move Z to target at a fixed starting Y (avoid snap).
        y_start = max(120.0f, goalY * 0.55f);
        if (y_start > goalY) y_start = goalY;

        // Phase 0: Ease into the plunge start pose (prevents violent snap on first write_sew_for()).
        {
          float tb0, ts0, te0, tw0;
          compute_arm_us(0.0f, y_start, GRAB_MAX_Z_MM, GRAB_GRIP_ANGLE_DEG, &tb0, &ts0, &te0, &tw0);
          int s0 = g_lastShoulderUs;
          int e0 = g_lastElbowUs;
          int w0 = g_lastWristPitchUs;
          int sT = (int)ts0;
          int eT = (int)te0;
          int wT = (int)tw0;
          uint16_t easeSteps = 64;
          for (uint16_t i = 1; i <= easeSteps; i++) {
            int rc = grab_check_serial(tX, tY, tZ);
            if (rc == 2) { got_clamp = true; break; }
            if (rc == 3) { aborted = true; break; }
            float lin = (float)i / (float)easeSteps;
            float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
            pwm.writeMicroseconds(baseMotor, lockedBaseUs);
            pwm.writeMicroseconds(shoulderMotor, (int)(s0 + s * (sT - s0)));
            pwm.writeMicroseconds(elbowMotor, (int)(e0 + s * (eT - e0)));
            pwm.writeMicroseconds(wristPitchMotor, (int)(w0 + s * (wT - w0)));
            delay(14);
          }
          // Sync last commanded pulses after ease-in
          g_lastBaseUs = lockedBaseUs;
          g_lastShoulderUs = sT;
          g_lastElbowUs = eT;
          g_lastWristPitchUs = wT;
        }
        current_grab_y = y_start;

        // Z descent: balanced speed vs smoothness (faster than V4 to reduce time off-target).
        uint16_t zSteps = 200;
        float zStart = GRAB_MAX_Z_MM;
        for (uint16_t i = 1; i <= zSteps; i++) {
          int rc = grab_check_serial(tX, tY, tZ);
          if (rc == 2) { got_clamp = true; break; }
          if (rc == 3) { aborted = true; break; }
          float lin = (float)i / (float)zSteps;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          float zCur = zStart + s * (tZ - zStart);
          write_sew_for(y_start, zCur);
          delay(16);
        }
      }

      if (aborted) {
        Serial.println("GRAB_ABORTED");
        return_to_search_pose(90, 14);
        lastCommandTime = millis();
        return;
      }

      // 1. OPEN GRIPPER (after Z is set)
      pwm.setPWM(gripperMotor, 0, 4096);
      for (int p = 1400; p >= GRIPPER_OPEN_US; p -= 10) {
        pwm.writeMicroseconds(gripperMotor, p);
        delay(15);
      }
      delay(150);

      // 2. FORWARD GLIDE: cosine-eased (slow start, cruise, slow finish) for smoothness.
      if (!got_clamp && !aborted) {
        float yFrom = y_start;
        if (yFrom > goalY) yFrom = goalY;
        float travel = goalY - yFrom;
        float finalReachedY = yFrom;
        if (travel > 0.25f) {
          // ~1.2mm average step -> enough steps to keep each servo write small.
          uint16_t ySteps = (uint16_t)(travel / 1.2f);
          if (ySteps < 30) ySteps = 30;
          if (ySteps > 260) ySteps = 260;
          for (uint16_t i = 1; i <= ySteps; i++) {
            int rc = grab_check_serial(tX, tY, tZ);
            if (rc == 2) { got_clamp = true; break; }
            if (rc == 3) { aborted = true; break; }
            float lin = (float)i / (float)ySteps;
            float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
            float y = yFrom + s * travel;
            write_sew_for(y, tZ);
            finalReachedY = y;
            delay(13);
          }
        }
        if (!got_clamp && !aborted && finalReachedY < goalY - 0.05f) {
          write_sew_for(goalY, tZ);
          finalReachedY = goalY;
        }
        goalY = finalReachedY;
      }

      if (aborted) {
        Serial.println("GRAB_ABORTED");
        return_to_search_pose(90, 14);
        lastCommandTime = millis();
        return;
      }

      // 3. If CLAMP hasn't arrived yet, hold position and keep accepting commands
      if (!got_clamp) {
        Serial.println("GRAB_AWAITING_CLAMP");
        uint32_t clamp_wait_start = millis();
        while (!got_clamp && (millis() - clamp_wait_start < 10000)) {
          refresh_arm_hold_pose();
          int rc = grab_check_serial(tX, tY, tZ);
          if (rc == 2) {
            got_clamp = true;
          } else if (rc == 3) {
            aborted = true;
            break;
          }
          delay(20);
        }
        if (aborted) {
          Serial.println("GRAB_ABORTED");
          return_to_search_pose(90, 14);
          lastCommandTime = millis();
          return;
        }
        if (!got_clamp) {
          Serial.println("CLAMP_TIMEOUT");
        }
      }

      delay(120);

      // 4b. PRE-CLAMP FORWARD NUDGE: push ~1.5 inches (38mm) farther forward
      //     RIGHT before the grab (cosine-eased for smoothness). This is the
      //     only "reach farther" change in this pass.
      if (!aborted) {
        const float nudgeMM = 38.0f;
        const float yFromNudge = current_grab_y;
        const float yToNudge   = yFromNudge + nudgeMM;
        const uint16_t nudgeSteps = 36;
        for (uint16_t i = 1; i <= nudgeSteps; i++) {
          int rc = grab_check_serial(tX, tY, tZ);
          if (rc == 3) { aborted = true; break; }
          float lin = (float)i / (float)nudgeSteps;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          float yCur = yFromNudge + s * (yToNudge - yFromNudge);
          write_sew_for(yCur, tZ);
          delay(13);
        }
      }

      if (aborted) {
        Serial.println("GRAB_ABORTED");
        return_to_search_pose(90, 14);
        lastCommandTime = millis();
        return;
      }

      // 5. CLAMP (close faster and firmly)
      for (int p = GRIPPER_OPEN_US; p <= GRIPPER_CLOSE_US; p += 20) {
        refresh_arm_hold_pose();
        pwm.writeMicroseconds(gripperMotor, p);
        delay(8);
      }
      {
        uint32_t t0 = millis();
        while (millis() - t0 < GRIPPER_HOLD_MS) {
          refresh_arm_hold_pose();
          delay(20);
        }
      }

      // 6. LIFT OFF (cosine-eased; use actual Y — not goalY — so early CLAMP does
      //    not jab forward on lift).
      {
        const float zFrom = tZ;
        const float zTo   = tZ + 100.0f;
        const uint16_t liftSteps = 80;
        const float liftYHold = current_grab_y;
        for (uint16_t i = 1; i <= liftSteps; i++) {
          float lin = (float)i / (float)liftSteps;
          float s = (1.0f - cosf(lin * 3.14159265f)) * 0.5f;
          float zCur = zFrom + s * (zTo - zFrom);
          write_sew_for(liftYHold, zCur);
          delay(14);
        }
      }

      // After grabbing: smoothly rotate to the DROP pose (base = 0°, all other
      // joints at calibrated home) and open the gripper to release the object.
      // This function is the one-and-only post-grab motion — a second
      // go_home_pose_smooth() here was redundant (~1.3s wait with no movement) and
      // could make it LOOK like the base never rotated to drop-off.
      return_to_home_drop_pose(90, 14);
      g_sweep_enabled = false;
      g_face_search_mode = false;

      Serial.println("GRAB_COMPLETE");
      lastCommandTime = millis();
    }
  }

  // 2. AUTO-SWEEP (disabled until a GRAB starts)
  if (g_sweep_enabled && (millis() - lastCommandTime > 1000)) {
    sweepOffset += sweepStep;
    if (sweepOffset >= sweepRange) {
      sweepOffset = sweepRange;
      sweepStep = -3; 
    } else if (sweepOffset <= -sweepRange) {
      sweepOffset = -sweepRange;
      sweepStep = 3;  
    }
    currentBasePulse = baseHome + sweepOffset;
    
    nodAngle += nodSpeed;
    if (nodAngle > PI * 2) { nodAngle -= PI * 2; }
    if (g_face_search_mode) {
      // Face-search: pitch sweeps UP from home (pitchHome is 90°/horizontal) and back
      // to 90° — opposite of the object-sweep nod, which dips below horizontal.
      // (1 - cos)/2 eases 0..1..0 across a full nodAngle cycle.
      float upFrac = (1.0f - cosf(nodAngle)) * 0.5f;
      currentPitchPulse = pitchHome - (int)(upFrac * (float)pitchUpDepth);
    } else {
      currentPitchPulse = pitchHome + (sin(nodAngle) * pitchDepth);
    }

    pwm.writeMicroseconds(baseMotor, currentBasePulse);
    pwm.writeMicroseconds(wristPitchMotor, currentPitchPulse);

    g_lastBaseUs = currentBasePulse;
    g_lastWristPitchUs = currentPitchPulse;

    delay(20); 
  }
}

void calculate_real_coords(float dist) {
  dist *= COORDS_DIST_SCALE;
  dist -= COORDS_TOF_TO_GRIPPER_TIP_MM;
  if (dist < 1.0f) {
    dist = 1.0f;
  }

  float baseAngleDeg = (baseHome - currentBasePulse) / 11.11;
  float baseAngleRad = baseAngleDeg * (PI / 180.0);
  float pitchAngleDeg = COORDS_PITCH_SIGN * ((pitchHome - currentPitchPulse) / 7.41);
  float pitchAngleRad = pitchAngleDeg * (PI / 180.0);

  float pivot_Y = 93.0;  
  float pivot_Z = 243.0; 
  float lens_forward = 25.0; 
  float lens_up = 16.5;      

  float current_lens_Y = pivot_Y + (lens_forward * cos(pitchAngleRad)) + (lens_up * sin(pitchAngleRad));
  float current_lens_Z = pivot_Z - (lens_forward * sin(pitchAngleRad)) + (lens_up * cos(pitchAngleRad));

  float projected_Y = dist * cos(pitchAngleRad);
  float projected_Z = dist * sin(pitchAngleRad);
  float total_radius = current_lens_Y + projected_Y;

  float objX = total_radius * sin(baseAngleRad);
  float objY = total_radius * cos(baseAngleRad);
  float rawZ = current_lens_Z + (COORDS_Z_SIGN * projected_Z) + COORDS_Z_OFFSET_MM;
  float objZ = pivot_Z + (rawZ - pivot_Z) * COORDS_Z_GAIN;

  Serial.print("COORDS_DBG rawZ=");
  Serial.print(rawZ);
  Serial.print(" objZ=");
  Serial.print(objZ);
  Serial.print(" pitch=");
  Serial.println(pitchAngleDeg);

  Serial.print("COORDS ");
  Serial.print(objX); Serial.print(",");
  Serial.print(objY); Serial.print(",");
  Serial.println(objZ);
}

String getValue(String data, char separator, int index) {
  int found = 0, strIndex[] = {0, -1}, maxIndex = data.length() - 1;
  for (int i = 0; i <= maxIndex && found <= index; i++) {
    if (data.charAt(i) == separator || i == maxIndex) {
      found++;
      strIndex[0] = strIndex[1] + 1;
      strIndex[1] = (i == maxIndex) ? i + 1 : i;
    }
  }
  return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}
