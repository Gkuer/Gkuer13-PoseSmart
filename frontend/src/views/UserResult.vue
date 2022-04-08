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
      <div class="content" :style="{ top: '20%' }">
        <div>활동 내역</div>
      </div>
    </div>
    <div class="table">
      <v-data-table
        :headers="headers"
        :items="results"
        :page.sync="page"
        :items-per-page="itemsPerPage"
        hide-default-footer
        class="elevation-1"
        @page-count="pageCount = $event"
      ></v-data-table>
      <div class="text-center pt-2">
        <v-pagination v-model="page" :length="pageCount"></v-pagination>
      </div>
      <v-btn color="error" class="del" @click="del">회원탈퇴</v-btn>
    </div>
  </div>
</template>

<script lang="ts">
import Vue from "vue";
import { getReports, deleteUser } from "@/api/user";
import jwt_decode from "jwt-decode";
interface Report {
  date: string;
  start_time: string;
  end_time: string;
  studyTime: string;
  neck_cnt: string;
  blink_cnt: string;
  stretching_cnt: string;
}
export default Vue.extend({
  data() {
    return {
      page: 1,
      pageCount: 0,
      itemsPerPage: 10,
      headers: [
        {
          text: "날짜",
          value: "date",
        },
        { text: "시작 시간", value: "start_time" },
        { text: "종료 시간 ", value: "end_time" },
        { text: "공부 시간", value: "study_time" },
        { text: "거북목 횟수", value: "neck_cnt" },
        { text: "깜빡임 횟수", value: "blink_cnt" },
        { text: "스트레칭 횟수", value: "stretching_cnt" },
      ],
      results: [],
    };
  },
  async created() {
    try {
      const data = await getReports();
      const res = data.data;
      res.forEach((data: any) => {
        const start_time: string = data.start_time;
        const end_time: string = data.end_time;
        const duringMsec: number =
          Date.parse(end_time) - Date.parse(start_time);
        const duringSec: number = duringMsec / 1000;
        const duringMin: number = duringMsec / 1000 / 60;
        const duringHour: number = duringMsec / 1000 / 60 / 60;
        const item = {
          start_time:
            data.start_time.substring(0, 10) +
            " " +
            data.start_time.substring(11, 19),
          end_time:
            data.end_time.substring(0, 10) +
            " " +
            data.end_time.substring(11, 19),
          neck_cnt: data.neck_cnt,
          blink_cnt: data.blink_cnt,
          stretching_cnt: data.stretching_cnt,
          date: data.start_time.substring(0, 10),
          study_time:
            duringHour.toFixed(0) +
            ":" +
            duringMin.toFixed(0) +
            ":" +
            duringSec.toFixed(0),
        } as never;
        this.results.push(item);
      });
    } catch (e) {
      console.log(e);
    }
  },
  methods: {
    home() {
      this.$router.push({ name: "LoginHome" });
    },
    logout() {
      sessionStorage.removeItem("accessToken");
      this.$router.push({ name: "HomeView" });
      this.$router.go(0);
    },
    del() {
      const token: any = jwt_decode(
        sessionStorage.getItem("accessToken") || "{}"
      );
      console.log(token);
      const userid = token.user_id;
      console.log(userid, "userid");
      try {
        deleteUser(userid);
        sessionStorage.removeItem("accessToken");
        this.$router.push({ name: "HomewView" });
        this.$router.go(0);
      } catch (error) {
        alert("회원 탈퇴 중 오류 발생");
      }
    },
  },
});
</script>

<style scoped>
.table {
  position: absolute;
  top: 55%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 5rem;
  color: black;
  font-weight: bold;
  z-index: 2;
  text-align: center;
  width: 1200px;
}
.del {
  font-size: 15px;
  font-weight: bold;
  left: 45%;
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
