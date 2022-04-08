import Vue from "vue";
import VueRouter, { RouteConfig } from "vue-router";
import HomeView from "../views/HomeView.vue";
import LoginHome from "../views/LoginHome.vue";
import DetectPose from "../views/DetectPose.vue";
import UserResult from "../views/UserResult.vue";

Vue.use(VueRouter);

const onlyAuthUser = async (to: any, from: any, next: any) => {
  // console.log(store);
  const token = sessionStorage.getItem("accessToken");
  console.log(token);
  if (token) {
    console.log("로그인 했다.");
    next();
  } else {
    alert("로그인이 필요한 페이지입니다..");
    router.push({ name: "Home" });
  }
};

const routes: Array<RouteConfig> = [
  {
    path: "/",
    name: "Home",
    component: HomeView,
  },
  {
    path: "/login",
    name: "LoginHome",
    component: LoginHome,
  },
  {
    path: "/detect",
    name: "DectectPose",
    beforeEnter: onlyAuthUser,
    component: DetectPose,
  },
  {
    path: "/result",
    name: "UserResult",
    beforeEnter: onlyAuthUser,
    component: UserResult,
  },
];

const router = new VueRouter({
  mode: "history",
  base: process.env.BASE_URL,
  routes,
});

export default router;
