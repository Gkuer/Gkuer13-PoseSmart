<template>
  <div>
    <div
      class="mainImg"
      :style="{ backgroundImage: `url(${require('@/assets/transmain.png')})` }"
    >
      <div class="logo">
        <img
          @click="home"
          src="@/assets/logo_transparent.png"
          alt=""
          width="300"
        />
      </div>
      <v-btn class="logout" @click="logout">LOGOUT</v-btn>
      <div class="content">
        <div>Time : {{ timeString }}</div>
        <div>
          <video
            autoplay
            :style="{ width: '500px', height: '500px', background: 'black' }"
          ></video>
        </div>

        <v-row>
          <v-col
            ><img
              v-if="mode === 'play'"
              class="icon"
              src="@/assets/play.png"
              alt=""
              @click="play()"
            />
            <img
              v-if="mode === 'pause'"
              @click="pause()"
              class="icon"
              src="@/assets/pause.png"
              alt=""
            />
            <img
              v-if="mode === 'ready'"
              @click="ready()"
              class="icon"
              src="@/assets/video.png"
              alt=""
            />
          </v-col>
          <v-col
            ><img class="icon" src="@/assets/exit.png" alt="" @click="exit()"
          /></v-col>
        </v-row>
      </div>
    </div>
    <!-- <a href="#">download</a> -->
  </div>
</template>

<script>
import Vue from "vue";
import { insertReports, getDetect, getDetectBlink, getTime } from "@/api/user";
export default Vue.extend({
  data() {
    return {
      localStream: {},
      localVideo: {},
      time: 0,
      timeString: "",
      timeset: {},
      mode: "ready",
      blink_cnt: 0,
      neck_cnt: 0,
      stretching_cnt: 0,
      start_time: "",
      isStart: false,
      imageCapture: {},
      file: null,
      face_x_mean: 0.0,
      face_y_mean: 0.0,
      nose_mean: 0.0,
      data: {
        blob_data: "",
        face_x_mean: 0.0,
        face_y_mean: 0.0,
        nose_mean: 0.0,
        face_x: "",
        face_y: "",
        nose_to_center: "",
        cnt: 0,
        flag: "",
      },
      resBlinkData: {
        blob_data: "",
        count: 0, // 변수 빼기
        total: 0,
        time: 0,
        flag: "",
      },
      overFive_data: {
        blob_data: "",
        face_x_mean: 0.0,
        face_y_mean: 0.0,
        nose_mean: 0.0,
        face_x: "",
        face_y: "",
        nose_to_center: "",
        cnt: 0,
        flag: "",
      },
      eyeTimeSet: "",
      userSetting: {},
      eyeAlarm: true,
      xCnt: 0,
      yCnt: 0,
      angleSound: {}, // 고개 사운드
      distanceSound: {}, // 거리 사운드
      eyeSound: {}, // 눈깜빡임 사운드
      notDetection: 0,
      isDetect: false,
      working: true,
    };
  },
  async created() {
    const data = await getTime();
    console.log(data.data);
    this.userSetting = data.data;
    console.log(this.userSetting);
    switch (this.userSetting.alarm_sound) {
      case 1:
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.notDetectionSound = new Audio(require("../assets/1감지.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.angleSound = new Audio(require("../assets/1목옆.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.distanceSound = new Audio(require("../assets/1목앞.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.eyeSound = new Audio(require("../assets/1눈.mp3"));
        break;
      case 2:
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.notDetectionSound = new Audio(require("../assets/2감지.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.angleSound = new Audio(require("../assets/2목옆.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.distanceSound = new Audio(require("../assets/2목앞.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.eyeSound = new Audio(require("../assets/2눈.mp3"));
        break;
      case 3:
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.notDetectionSound = new Audio(require("../assets/3감지.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.angleSound = new Audio(require("../assets/3목옆.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.distanceSound = new Audio(require("../assets/3목앞.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.eyeSound = new Audio(require("../assets/3눈.mp3"));
        break;
      case 4:
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.notDetectionSound = new Audio(require("../assets/4감지.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.angleSound = new Audio(require("../assets/4목옆.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.distanceSound = new Audio(require("../assets/4목앞.mp3"));
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        this.eyeSound = new Audio(require("../assets/4눈.mp3"));
        break;
    }
  },
  methods: {
    ready() {
      // 알림음 정자세를 유지해주세요(4초에서 ~ 10초 정도) 감지오류
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      let detectSound = new Audio(require("../assets/감지.mp3")); // 4초 감지 사운드로 변경해야함
      detectSound.play();
      this.isDetect = !this.isDetect;
      this.localVideo = document.querySelector("video");
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((mediaStream) => {
          this.localStream = mediaStream;
          this.localVideo.srcObject = mediaStream;
          const track = mediaStream.getVideoTracks()[0];
          this.imageCapture = new ImageCapture(track);
        })
        .catch((e) => console.log(e));

      let timeSet = setInterval(() => {
        this.imageCapture.takePhoto().then((blob) => {
          const reader = new FileReader();
          reader.onload = async () => {
            try {
              //TODO
              this.data.blob_data = reader.result;
              if (this.data.cnt == 4) {
                this.overFive_data.cnt = this.data.cnt;
                this.nose_mean = this.data.nose_mean;
                this.face_x_mean = this.data.face_x_mean;
                this.face_y_mean = this.data.face_y_mean;
                this.overFive_data.face_x_mean = this.face_x_mean;
                this.overFive_data.face_y_mean = this.face_y_mean;
                this.overFive_data.nose_mean = this.nose_mean;
                this.overFive_data.cnt = this.data.cnt;
                this.mode = "play";
                console.log("초기세팅끝", this.mode);
                this.localVideo.srcObject = null;
                clearInterval(timeSet);
                return;
              }
              this.data = (await getDetect(this.data)).data;
              console.log("less Cnt 4 response", this.data.cnt);
            } catch (error) {
              console.log(error);
            }
          };
          reader.readAsDataURL(blob);
        });
      }, 1000);
    },
    home() {
      this.$router.push({ name: "LoginHome" });
    },
    logout() {
      sessionStorage.removeItem("accessToken");
      this.$router.push({ name: "HomeView" });
      this.$router.go(0);
    },
    pause() {
      if (this.time > 0) {
        clearInterval(this.eyeTimeSet);
        clearInterval(this.timeset.eyeTimeSet);
        clearInterval(this.timeset);
        clearInterval(this.neckTime);
        this.mode = "play";
        this.localVideo.srcObject = null;
      }
    },
    async exit() {
      // 나갈때 로그인 홈으로 보내기전에 reports를 저장
      if (this.isStart) {
        try {
          await insertReports({
            blink_cnt: this.blink_cnt,
            neck_cnt: this.neck_cnt,
            stretching_cnt: this.stretching_cnt,
            start_time: this.start_time,
            end_time: new Date(),
          });
        } catch (error) {
          console.log(error);
        }
      }
      clearInterval(this.timeset);
      clearInterval(this.eyeTimeSet);
      clearInterval(this.neckTime);
      this.$router.push({ name: "LoginHome" });
    },
    takePhoto() {
      this.imageCapture.takePhoto().then((blob) => {
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            //TODO
            this.data.blob_data = reader.result;
            this.overFive_data.blob_data = reader.result;
            this.overFive_data.cnt = this.overFive_data.cnt + 1;
            const data = (await getDetect(this.overFive_data)).data;
            console.log("Over 4 res", data);
            if (data.detection_flag === "false") {
              if (++this.notDetection % 3 === 0) {
                this.notDetectionSound.play();
                this.working = false;
              }
            }
            if (data.detection_flag === "detected") {
              this.working = true;
              if (!data.x_result) {
                this.xCnt++; //  고개 기울어진 횟수
              }
              if (!data.y_result) {
                this.yCnt++; // 모니터랑 가까워진 횟수
              }
              if (this.time % this.userSetting.neck_time === 0) {
                if (
                  this.yCnt >= this.userSetting.neck_time / 2 &&
                  this.xCnt >= this.userSetting.neck_time / 2
                ) {
                  this.distanceSound.play();
                  this.distanceSound.onended = () => {
                    this.angleSound.play();
                  };
                } else if (this.yCnt >= this.userSetting.neck_time / 2) {
                  this.distanceSound.play();
                } else if (this.xCnt >= this.userSetting.neck_time / 2) {
                  this.angleSound.play();
                }
                this.yCnt = 0;
                this.xCnt = 0;
                this.neck_cnt++;
              }
            }
          } catch (error) {
            console.log(error, "1");
          }
        };
        reader.readAsDataURL(blob);
      });
    },
    play() {
      const reader = new FileReader();
      reader.onload = async () => {
        try {
          //TODO
          this.resBlinkData.blob_data = reader.result;
          this.resBlinkData = (await getDetectBlink(this.resBlinkData)).data;
          if (this.resBlinkData.res === true) {
            // 알림음 울리기

            this.eyeAlarm = true;
            clearInterval(this.eyeTimeSet);
          } // 20초 안에 true나오면 그만 보냈다가 다시 20초 되면 보내기
          console.log("res : ", this.resBlinkData);
        } catch (error) {
          console.log(error);
        }
      };

      // 여기서 모델 탐지하는 통신을 해야함
      if (!this.isStart) {
        this.start_time = new Date();
        this.isStart = true;
      }
      this.timeset = setInterval(() => {
        this.time += 1;
        let hour = parseInt(this.time / 3600);
        let min = parseInt((this.time % 3600) / 60);
        let second = parseInt(this.time % 60);
        this.timeString = hour + " : " + min + " : " + second;
        if (second % this.userSetting.blink_time === 0) {
          if (!this.eyeAlarm) {
            // 알람 울리기
            if (this.working) {
              this.blink_cnt++;
              this.eyeSound.play();
            }
          }
          this.eyeAlarm = false;
          this.eyeTimeSet = setInterval(() => {
            console.log("eyeTime");
            this.imageCapture.takePhoto().then((blob) => {
              reader.readAsDataURL(blob);
            });
          }, 500); // 0 -> 10 +10 / 20
        }

        if (this.time % this.userSetting.stretching_time === 0) {
          // 스트레칭 알림음 주기
          this.stretching_cnt++;
          window.open(
            "https://youtu.be/fmGibtjyy5o",
            "",
            "width=800, height=600"
          );
        }
      }, 1000);

      this.neckTime = setInterval(() => {
        this.takePhoto();
      }, 1000);

      this.localVideo = document.querySelector("video");
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((mediaStream) => {
          this.localStream = mediaStream;
          this.localVideo.srcObject = mediaStream;
          const track = mediaStream.getVideoTracks()[0];
          this.imageCapture = new ImageCapture(track);
        })
        .catch((e) => console.log(e));

      this.mode = "pause";
    },
  },
});
</script>

<style scoped>
.icon {
  width: 150px;
  height: 150px;
}
.content {
  top: 45%;
}
.logo {
  position: absolute;
  font-size: 3rem;
  color: black;
  font-weight: bold;
  z-index: 2;
  left: -2%;
  top: -7.2%;
}
</style>
