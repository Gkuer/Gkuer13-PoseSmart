<template>
  <div>
    <div
      class="mainImg"
      :style="{ backgroundImage: `url(${require('@/assets/transmain.png')})` }"
    >
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
              v-if="!playing"
              class="icon"
              src="@/assets/play.png"
              alt=""
              @click="play()"
            />
            <img
              v-if="playing"
              @click="pause()"
              class="icon"
              src="@/assets/pause.png"
              alt=""
            />
          </v-col>
          <v-col></v-col>
          <v-col
            ><img class="icon" src="@/assets/exit.png" alt="" @click="exit()"
          /></v-col>
        </v-row>
      </div>
    </div>
  </div>
</template>

<script>
import Vue from "vue";
import { insertReports } from "@/api/user";
export default Vue.extend({
  data() {
    return {
      localStream: {},
      localVideo: {},
      time: 0,
      timeString: "",
      timeset: {},
      playing: false,
      blink_cnt: 0,
      neck_cnt: 0,
      stretching_cnt: 0,
      start_time: "",
      isStart: false,
    };
  },
  methods: {
    pause() {
      if (this.time > 0) {
        clearInterval(this.timeset);
        this.playing = !this.playing;
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
      this.$router.push({ name: "LoginHome" });
    },
    play() {
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
      }, 1000);
      this.localVideo = document.querySelector("video");
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((mediaStream) => {
          this.localStream = mediaStream;
          this.localVideo.srcObject = mediaStream;
        })
        .catch((e) => console.log(e));
      this.playing = true;
    },
  },
  // mounted() {
  //   this.init();
  // },
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
</style>
