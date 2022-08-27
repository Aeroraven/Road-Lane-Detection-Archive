import { createApp } from "vue";
import { createPinia } from "pinia";
import Particles from "particles.vue3";
import App from "./App.vue";
import router from "./router";
import naive from 'naive-ui'
import { Tensor } from 'onnxruntime-web'
import 'animate.css'

//import {cvWrapper} from './utils/opencv.js'

const app = createApp(App);

//app.config.globalProperties.$cv = cvWrapper

app.use(naive)
app.use(createPinia());
app.use(router);
app.use(Particles);
app.mount("#app");


