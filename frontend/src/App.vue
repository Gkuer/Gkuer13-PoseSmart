<template>
  <v-app>
    <v-main>
      <router-view />
    </v-main>
  </v-app>
</template>

<script lang="ts">
import Vue from "vue";

export default Vue.extend({
  name: "App",

  data: () => ({
    //
  }),
});
</script>
<style scope>
@import url("https://fonts.googleapis.com/css2?family=Jua&display=swap");
* {
  font-family: "Jua", sans-serif;
}
.logo {
  position: absolute;
  font-size: 3rem;
  color: black;
  font-weight: bold;
  z-index: 2;
  left: -2%;
  top: -12%;
}
.mainImg {
  position: relative;
  height: 100vh;
  background-size: cover;
}
.loginImg {
  position: relative;
  height: 60vh;
  background-size: cover;
}
.content {
  position: absolute;
  top: 40%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 5rem;
  color: black;
  font-weight: bold;
  z-index: 2;
  text-align: center;
  width: 750px;
}
</style>
