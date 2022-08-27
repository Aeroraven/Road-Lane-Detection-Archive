import { createRouter, createWebHashHistory } from "vue-router";
import HomeView from "../views/HomeView.vue";
import ImageView from "../views/ImageView.vue";
import VideoView from "../views/VideoView.vue";
import ARView from '../views/ARView.vue'
import TestView from '../views/TestView.vue'


const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    {
      path: "/",
      name: "home",
      component: HomeView,
    },
    {
      path: "/image",
      name: "imageview",
      component: ImageView,
    },
    {
      path: "/video",
      name: "videoview",
      component: VideoView,
    },
    {
      path: "/ar",
      name: "arview",
      component: ARView,
    },
    {
      path: "/help",
      name: "testview",
      component: TestView,
    },
  ],
});

export default router;
