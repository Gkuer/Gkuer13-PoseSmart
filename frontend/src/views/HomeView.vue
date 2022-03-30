<template>
  <div>
    <div
      class="mainImg"
      :style="{ backgroundImage: `url(${require('@/assets/main.jpg')})` }"
    >
      <div v-if="mode === 'home'">
        <div class="content">
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <v-row :style="{ marginLeft: '200px' }">PoseSmart</v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15" :style="{ fontSize: '4rem' }"
            >AI 거북목 & 눈 깜빡임 감지</v-row
          >
          <v-row class="mt-8 ml-15" :style="{ fontSize: '3rem' }"
            >간단히 가입하고 이용하세요</v-row
          >
          <v-row class="mt-10">
            <v-col></v-col>
            <v-col
              ><v-btn
                :style="{ width: '15rem', height: '3rem', fontSize: '1.5rem' }"
                color="warning"
                @click="setMode('login')"
                >로그인</v-btn
              ></v-col
            >
            <v-col></v-col>
            <v-col></v-col>
            <v-col
              ><v-btn
                :style="{ width: '15rem', height: '3rem', fontSize: '1.5rem' }"
                color="warning"
                @click="setMode('register')"
                >회원 가입
              </v-btn></v-col
            >
            <v-col></v-col>
            <v-col></v-col>
          </v-row>
          <v-row class="mt-15"></v-row>
        </div>
      </div>
      <div v-if="mode === 'login'">
        <div class="logo">PoseSmart</div>
        <div class="content">
          <v-row class="mt-15"></v-row>
          <v-row :style="{ marginLeft: '250px' }">로그인</v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <div
            :style="{ marginTop: '50px', marginLeft: '120px', width: '500px' }"
          >
            <v-text-field
              :style="{ fontSize: '1.5rem' }"
              label="아이디"
              v-model="id"
            ></v-text-field>
            <v-text-field
              :style="{ fontSize: '1.5rem' }"
              label="비밀번호"
              v-model="password"
            ></v-text-field>
          </div>
          <v-row class="mt-5">
            <v-col></v-col>
            <v-col
              ><v-btn
                :style="{ width: '15rem', height: '3rem', fontSize: '1.5rem' }"
                color="warning"
                @click="login"
                >로그인</v-btn
              ></v-col
            >
            <v-col></v-col>
          </v-row>
        </div>
      </div>
      <div v-if="mode === 'register'">
        <div class="logo">PoseSmart</div>
        <div class="content">
          <v-row class="mt-15"></v-row>
          <v-row :style="{ marginLeft: '230px' }">회원 가입</v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <v-row class="mt-15"></v-row>
          <div :style="{ marginLeft: '120px', width: '500px' }">
            <v-text-field label="아이디" v-model="registerId"></v-text-field>
            <v-text-field
              label="비밀번호"
              v-model="registerPassword"
            ></v-text-field>
          </div>
          <v-row class="mt-5">
            <v-col></v-col>
            <v-col
              ><v-btn
                :style="{ width: '15rem', height: '3rem', fontSize: '1.5rem' }"
                color="warning"
                @click="register"
                >회원 가입</v-btn
              ></v-col
            >
            <v-col></v-col>
          </v-row>
        </div>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
import Vue from "vue";
import { registerUser, loginUser } from "@/api/user";
export default Vue.extend({
  name: "HomePage",
  methods: {
    async register(): Promise<void> {
      console.log({
        username: this.registerId,
        password: this.registerPassword,
      });
      let regEmail =
        /^[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*@[0-9a-zA-Z]([-_.]?[0-9a-zA-Z])*.[a-zA-Z]{2,3}$/i;

      if (regEmail.test(this.registerId) === false) {
        alert("이메일 형식으로 입력해주세요");
        return;
      }
      try {
        await registerUser({
          username: this.registerId,
          password: this.registerPassword,
        });
        location.reload();
      } catch (error) {
        alert("로그인 불가");
      }
    },
    setMode(val: string): void {
      this.mode = val;
    },
    async login(): Promise<void> {
      console.log({ username: this.id, password: this.password });
      try {
        const data = await loginUser({
          username: this.id,
          password: this.password,
        });
        sessionStorage.setItem("accessToken", data.data.access);
        this.$router.push({ name: "LoginHome" });
      } catch (error) {
        alert("로그인 할 수 없습니다. 아이디나 비밀번호를 확인해주세요.");
      }
    },
  },
  data() {
    return {
      mode: "home",
      id: "",
      password: "",
      registerId: "",
      registerPassword: "",
    };
  },
});
</script>
<style scoped>
.v-text-field input {
  font-size: 1.2em;
}
</style>
