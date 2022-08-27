<script setup>
import {reactive,computed, ref, watch, onMounted,nextTick} from  'vue'
import { useRouter, onBeforeRouteUpdate } from "vue-router";
import { loadFull } from "tsparticles";
import { optionsMap } from './utils/ParticlePreset';
import Rosemary from './utils/Rosemary'
import router from './router'

function goHome(){
  router.push({path:"/"})
}
function goHelp(){
  router.push({path:"/help"})
}

let loadCurProg = 0
let loadTotalProg = 16
const loadProgress = ref(0)
const loadedStatus = ref(false)
async function particlesInit(engine) {
  await loadFull(engine);
  console.log("INIT")
}

const loadProgressComputed = computed(()=>{
  return Math.min(loadProgress.value,100)
});
const loadProgressComputedStyle = computed(()=>{
  return {
    'width': Math.min(loadProgress.value/2,50) + '%'
  }
});
const routerGlobal = reactive(useRouter());
const loadingDesc = ref("正在准备加载...")
const pageTitle = ref("未知")
const pageTitleEn = ref("Unknown")
const options = computed(()=> {
  console.log(optionsMap[optionSelected.value])
  return optionsMap[optionSelected.value];
})
watch(
  routerGlobal,
  ()=>{
    const o = routerGlobal.currentRoute.name;
    window.s = o
    console.log("Watch",routerGlobal.currentRoute.name)
    if(o == "videoview"){
      pageTitle.value = "视频识别"
      pageTitleEn.value = "Video Recognition"
    }
    if(o == "home"){
      pageTitle.value = "平台主页"
      pageTitleEn.value = "homepage"
    }
    if(o == "imageview"){
      pageTitle.value = "图像识别"
      pageTitleEn.value = "image recognition"
    }
    if(o == "arview"){
      pageTitle.value = "实时识别"
      pageTitleEn.value = "realtime recognition"
    }
    if(o == "testview"){
      pageTitle.value = "帮助"
      pageTitleEn.value = "help"
    }
  }
)
const optionSelected = ref("crazyParticles")

// Loading Progress Bar
const explData = ref(window.navigator.userAgent)
const rosemary = new Rosemary()
const successHandler = ()=>{
  loadCurProg+=1
  loadProgress.value = parseInt(Math.ceil(loadCurProg/loadTotalProg*100))
  if(loadCurProg==loadTotalProg&&loadedStatus.value==false){
    setTimeout(()=>{
        loadedStatus.value = true
    },1000) 
  }
}
const progressHandler = (x)=>{
  loadingDesc.value = x
}
onMounted(nextTick(async ()=>{
  await rosemary.rosLoader(successHandler,progressHandler)
}))

</script>

<template>

  <div class="alwen_main_wrapper">
        
    <Particles
      v-if="loadedStatus"
      id="tsparticles"
      :particlesInit="particlesInit"
      :options="options"
      :key="optionSelected"
    />
    <div class="alwen_svg_defs" style="display:none">
      <svg viewBox="0 0 122.88 106.43"><defs><path id="svg-def-rotation" fill="currentColor" d="M11.1 0h35.63c3.05 0 5.85 1.25 7.85 3.25 2.03 2.03 3.25 4.8 3.25 7.85v31.46h-3.19V12.18H3.15v75.26h7.61v11.61c0 1.58.27 3.1.77 4.51h-.43c-3.05 0-5.85-1.25-7.85-3.25C1.22 98.27 0 95.51 0 92.45V11.1c0-3.05 1.25-5.85 3.25-7.85C5.28 1.22 8.04 0 11.1 0zm83.85 33.45c-.37-5.8-2.64-10.56-6.06-13.97-3.64-3.63-8.59-5.74-13.94-5.93l2.46 2.95c.73.88.62 2.18-.26 2.91s-2.18.62-2.91-.26l-5.72-6.85a2.07 2.07 0 01.22-2.88l6.71-5.89c.86-.75 2.16-.66 2.91.19.75.86.66 2.16-.19 2.91l-3.16 2.78c6.43.21 12.4 2.75 16.8 7.13 4.07 4.06 6.79 9.69 7.25 16.49l2.58-3.08c.73-.88 2.04-.99 2.91-.26.88.73.99 2.04.26 2.91l-5.73 6.84c-.72.86-1.99.99-2.87.29l-6.98-5.56a2.077 2.077 0 01-.33-2.91c.71-.89 2.01-1.04 2.91-.33l3.14 2.52zm27.93 26.25v35.63c0 3.05-1.25 5.85-3.25 7.85-2.03 2.03-4.8 3.25-7.85 3.25h-78.9c-3.05 0-5.85-1.25-7.85-3.25-2.03-2.03-3.25-4.8-3.25-7.85V59.7c0-3.05 1.25-5.85 3.25-7.85 2.03-2.03 4.79-3.25 7.85-3.25h78.9c3.05 0 5.85 1.25 7.85 3.25 2.03 2.03 3.25 4.79 3.25 7.85zM35.41 77.49c0 2.51-2.03 4.57-4.57 4.57-2.51 0-4.57-2.03-4.57-4.57 0-2.51 2.03-4.57 4.57-4.57 2.52 0 4.57 2.03 4.57 4.57zm2.47-25.74v51.49h72.82V51.75H37.88z"></path></defs></svg>
    </div>
    <div class="alwen_ak_media_rotate" style="">
      <div class="alwen_ak_media_rotate_svg">
        <svg style="width:123px; height:107px;">
          <use xlink:href="#svg-def-rotation"></use>
        </svg> 
        <br/><br/>
        本应用不支持手机竖屏浏览，请旋转至横屏模式以继续浏览
      </div>
      
    </div>
    <transition name="fade" mode="out-in">
      <div class="alwen_ak_loading_wrapper" v-if="!loadedStatus">
        <div style="position:fixed;left:0px;bottom:0px;z-index:999;color:#666666;font-size:10px;font-family:sans-serif">
          应用处于开发阶段，不代表最终品质
          <br/>
          {{explData}}
        </div>
        <div class="alwen-phone-only">
          <div class="alwen_ak_loading_prog_logo-phone">
            车道线识别平台 <span class="alwen_ak_non_bold">Road Lane Detection</span><br/>
            <div class="alwen-skew-c alwen_ak_loading_prog_logo_sub" style="width:40%">
              <div style="position:relative;top:-2px">同济大学软件学院 专业方向综合项目(机器智能)</div>
            </div>
          </div>
        </div>
        <div class="alwen-pc-only">
          <div class="alwen_ak_loading_prog_logo">
            车道线识别平台 <span class="alwen_ak_non_bold">Road Lane Detection</span><br/>
            <div class="alwen-skew-c alwen_ak_loading_prog_logo_sub" style="width:40%">
              <div style="position:relative;top:-2px">同济大学软件学院 专业方向综合项目(机器智能)</div>
            </div>
          </div>
        </div>
        <div class="alwen-phone-only">
          <div class="alwen_ak_loading_prog_counter-phone">
            {{loadProgressComputed}}% <br/><br/>
            Now Loading...<br/><br/>
            <small style="font-family:'SourceHan',sans-serif;color:#888888">{{loadingDesc}}</small>
          </div>
        </div>
        <div class="alwen-pc-only">
          <div class="alwen_ak_loading_prog_counter">
            {{loadProgressComputed}}% <br/><br/>
            Now Loading...<br/><br/>
            <small style="font-family:'SourceHan',sans-serif;color:#888888">{{loadingDesc}}</small>
          </div>
        </div>
        
        
        <div class="alwen_ak_loading_progbar" :style="loadProgressComputedStyle">
          
        </div>
        <div class="alwen_ak_loading_progbar_r" :style="loadProgressComputedStyle">
          
        </div>
      </div>
    </transition>
    
    <transition name="fade" mode="out-in">
      <div v-if="loadedStatus">
        <div class="alwen_title" style="position:relative;z-index:99">
          <n-icon size="40" style="padding-right:20px;padding-left:20px">
            <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 640 512"><path d="M624 352h-16V243.9c0-12.7-5.1-24.9-14.1-33.9L494 110.1c-9-9-21.2-14.1-33.9-14.1H416V48c0-26.5-21.5-48-48-48H48C21.5 0 0 21.5 0 48v320c0 26.5 21.5 48 48 48h16c0 53 43 96 96 96s96-43 96-96h128c0 53 43 96 96 96s96-43 96-96h48c8.8 0 16-7.2 16-16v-32c0-8.8-7.2-16-16-16zM160 464c-26.5 0-48-21.5-48-48s21.5-48 48-48s48 21.5 48 48s-21.5 48-48 48zm320 0c-26.5 0-48-21.5-48-48s21.5-48 48-48s48 21.5 48 48s-21.5 48-48 48zm80-208H416V144h44.1l99.9 99.9V256z" fill="currentColor"></path></svg>
          </n-icon>
          <div style="display:inline-block">
            <span class="alwen_title_cn">
            车道线识别平台 <span class="alwen_ak_non_bold">Lane Detection</span>
            </span><br/>
            <span>
              专业方向综合设计项目(机器智能)
            </span>
          </div>
          <div style="position:fixed;right:10%;display:inline-block">
            <n-icon size="30" style="padding-right:20px;padding-left:20px;padding-top:10px;" @click="goHome">
              <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 576 512"><path d="M280.37 148.26L96 300.11V464a16 16 0 0 0 16 16l112.06-.29a16 16 0 0 0 15.92-16V368a16 16 0 0 1 16-16h64a16 16 0 0 1 16 16v95.64a16 16 0 0 0 16 16.05L464 480a16 16 0 0 0 16-16V300L295.67 148.26a12.19 12.19 0 0 0-15.3 0zM571.6 251.47L488 182.56V44.05a12 12 0 0 0-12-12h-56a12 12 0 0 0-12 12v72.61L318.47 43a48 48 0 0 0-61 0L4.34 251.47a12 12 0 0 0-1.6 16.9l25.5 31A12 12 0 0 0 45.15 301l235.22-193.74a12.19 12.19 0 0 1 15.3 0L530.9 301a12 12 0 0 0 16.9-1.6l25.5-31a12 12 0 0 0-1.7-16.93z" fill="currentColor"></path></svg>
            </n-icon>
            <n-icon size="30" style="padding-right:20px;padding-left:20px;padding-top:10px;"  @click="goHelp">
              <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 384 512"><path d="M202.021 0C122.202 0 70.503 32.703 29.914 91.026c-7.363 10.58-5.093 25.086 5.178 32.874l43.138 32.709c10.373 7.865 25.132 6.026 33.253-4.148c25.049-31.381 43.63-49.449 82.757-49.449c30.764 0 68.816 19.799 68.816 49.631c0 22.552-18.617 34.134-48.993 51.164c-35.423 19.86-82.299 44.576-82.299 106.405V320c0 13.255 10.745 24 24 24h72.471c13.255 0 24-10.745 24-24v-5.773c0-42.86 125.268-44.645 125.268-160.627C377.504 66.256 286.902 0 202.021 0zM192 373.459c-38.196 0-69.271 31.075-69.271 69.271c0 38.195 31.075 69.27 69.271 69.27s69.271-31.075 69.271-69.271s-31.075-69.27-69.271-69.27z" fill="currentColor"></path></svg>
            </n-icon>
          </div>
          
        </div>
        <router-view/>
        <div class="alwen-triangle-page-title-phone" style="position:fixed;bottom:0px;left:0px;">
          
          <span class="alwen-triangle-right-pr" style="">
            <div style="padding-top:10px;padding-left:20px;padding-right:20px;display:inline-block" >
              <b>{{pageTitle}}</b> <small>{{pageTitleEn}}</small>
            </div>
          </span>
          <span class="alwen-triangle-right" style="">

          </span>
          <span style="width:100%;background-color:#efefef;height:30px;display:inline-block;transform:translateX(-40px);z-index:0;position:absolute;bottom:0px;">
            
          </span>
        </div>
      </div>
    </transition>
  </div>
</template>

<style lang="scss">
$backcolor: #222222;

.alwen-triangle-page-title-phone{
  width:100%;
  left:0px;

}
//Mobile
@media screen and (max-width: 1020px) {
    .alwen_ak_loading_prog_logo{
      top:22% !important;
    }
    .alwen-menu-tip {
      display:none !important;
    }
    .alwen-pc-only {
      display:none !important;
    }
    .alwen-triangle-page-title{
      display:none !important;
    }
    .alwen-upload-pc{
      display:none !important
    }
    .alwen-menu-select-ar{
      margin-top:0px;
    }
}
//PC
@media screen and (min-width: 1020px) {
  
  .alwen-triangle-page-title-phone{
    display:none !important;
  }
  .alwen-upload-mobile{
    display:none !important
  }
  .alwen-phone-only {
    display:none !important;
  }
  .alwen-menu-select-ar{
      margin-top:60px;
    }
}
@media screen and (orientation: landscape) {
  .alwen_ak_media_rotate{
    display:none;
  }
}

.alwen_ak_non_bold{
  font-weight:normal;
}
.alwen_ak_media_rotate_svg{
  position:absolute;
  top: 50%;
  left:50%;
  transform: translate(-50%,-50%);
}
.alwen_ak_media_rotate{
  position:fixed;
  top:0px;
  bottom:0px;
  right:0px;
  left:0px;
  background-color: #efefefef;
  color:$backcolor;
  text-align: center;
  z-index: 11;
}
.fade-enter-active,
.fade-leave-active {
    transition: opacity 1s ease;
}

.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
.alwen_ak_loading_prog_logo_sub{
  font-family: 'Geometos','SourceHan',sans-serif;
  font-weight: bold;
}
.alwen_ak_loading_prog_logo{
  position:absolute;
  top: 28%;
  left:50%;
  transform: translate(-50%,-50%);
  height:5px;
  font-family: 'Geometos','SourceHanSerifHeavy',sans-serif;
  font-size:30px;
  font-weight: normal;
  text-align:center;
  width:100%;
}
.alwen_ak_loading_prog_logo-phone{
  position:absolute;
  top: 28%;
  left:50%;
  transform: translate(-50%,-50%);
  height:5px;
  font-family: 'Geometos','SourceHanSerifHeavy',sans-serif;
  font-size:24px;
  font-weight: normal;
  text-align:center;
  width:100%;
}
.alwen_ak_loading_prog_counter{
  position:absolute;
  top: 55%;
  left:50%;
  transform: translate(-50%,-50%);
  height:5px;
  font-family: 'Geometos','SourceHan',sans-serif;
  font-size:18px;
  font-weight: normal;
  text-align:center;
}
.alwen_ak_loading_prog_counter-phone{
  position:absolute;
  top: 55%;
  left:50%;
  transform: translate(-50%,-50%);
  height:5px;
  font-family: 'Geometos','SourceHan',sans-serif;
  font-size:16px;
  font-weight: normal;
  text-align:center;
}
.alwen_ak_loading_progbar{
  position:absolute;
  top: 50%;
  width: 40%;
  transform: translate(0%,-50%);
  height:5px;
  border-radius: 3px;
  background-color:#efefef;
  box-shadow:#aaaaaa 0px 0px 18px ;
  transition: all 0.5s;
}

.alwen_ak_loading_progbar_r{
  position:absolute;
  top: 50%;
  width: 40%;
  right:0px;
  transform: translate(0%,-50%);
  height:5px;
  border-radius: 3px;
  background-color:#efefef;
  box-shadow:#aaaaaa 0px 0px 18px ;
  transition: all 0.5s;

}
.alwen_ak_loading_wrapper{
  position:fixed;
  top:0px;
  bottom:0px;
  right:0px;
  left:0px;
  background-color: $backcolor;
  z-index: 10;
}
/*
@font-face{
  font-family:'Novecento';
  src:url('/fonts/Geometos.ttf');
}
@font-face{
  font-family:'Geometos';
  src:url('/fonts/Geometos.ttf');
}
@font-face{
  font-family:'SourceHan';
  src:url('/fonts/SourceHanSansCN-Normal.otf');
}
@font-face{
  font-family:'SourceHanSerif';
  src:url('/fonts/SourceHanSerifSC-Regular.otf');
}
@font-face{
  font-family:'SourceHanSerifHeavy';
  src:url('/fonts/SourceHanSerifSC-Heavy.otf');
}
@font-face{
  font-family:'Bender';
  src:url('/fonts/Bender.932867e7.ttf');
}*/
.alwen_main_wrapper{
  background-color: $backcolor;
  position: fixed;
  width:100%;
  height:100%;
  padding-left:0px;
  padding-top:5px;
  font-family: 'Novecento','SourceHan';
  font-weight: bold;
}
.alwen_title{
  margin-left:5%;
  margin-top:10px
}
.alwen_title_cn{
  font-size:20px;
}
.alwen-menu-select{
  position:absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%,-50%);
}
.alwen-menu-select-v{
  position:absolute;
  top: 50%;
  transform: translate(0,-50%);
}
.alwen-menu-alternative{
  text-align: center;
  width: 100%;
}
.alwen-menu-title{
  font-size:25px;
}
.alwen-menu-title-s{
  font-family:'Bender';
  font-weight: normal;
  font-size:20px;
}
.alwen-menu-select{
  width:100%
}
.alwen-skew {
    background-color: #efefef;
    width:55%;
    height: 30px;
    text-align: center;
    transform: skewX(-45deg);
    display: inline-block;
}

.alwen-skew>div {
    color: $backcolor;
    font-size:16px;
    padding-top:2px;
    transform: skewX(45deg);
}
.alwen-skew-b {
    text-align: center;
    transform: skewX(-45deg);
    display: inline-block;
}

.alwen-skew-b>div {
    font-size:16px;
    padding-top:2px;
    transform: skewX(45deg);
}
.alwen-menu-tip{
  margin-top:65px;
  text-align: center;
  width:100%;
  display: inline-block;
}
.alwen-number-counter{
  font-size:30px;
  position: fixed;
  padding-left:10px;
  color:#555555;
}

.alwen-skew-c {
    background-color: #efefef;
    width:55%;
    height: 20px;
    text-align: center;
    transform: skewX(-45deg);
    display: inline-block;
}
.alwen-skew-c>div {
    color: $backcolor;
    font-size:14px;
    padding-top:2px;
    transform: skewX(45deg);
}
body {
  -webkit-touch-callout: none; /* iOS Safari */
  -webkit-user-select: none; /* Chrome/Safari/Opera */
  -khtml-user-select: none; /* Konqueror */
  -moz-user-select: none; /* Firefox */
  -ms-user-select: none; /* Internet Explorer/Edge */
  user-select: none; 
}
.alwen-modal-title{
  font-family: 'Geometos','SourceHanSerifHeavy',sans-serif !important;
  color:#efefef;
}
.alwen-modal{
  font-family: 'Geometos','SourceHan',sans-serif !important;
  background-color: $backcolor;
  color: #efefef;
}
.alwen-triangle-right{
	display:inline-block;
	width:0;
	height:0;
	border-top: 25px solid transparent;
	border-left: 40px solid #efefef;
	border-bottom: 25px solid transparent;
  transform: translateY(17px);
}
.alwen-triangle-right-pr{
  background-color:#efefef;
  display:inline-block;
  height:50px;
  color: $backcolor;
  font-size:20px;
  margin-left:0px;
}
.alwen-bordered{
  aspect-ratio: 16/9;
  width:100%;
  border: #efefef 4px dashed;
  border-radius: 10px;
  text-align:center
}
.alwen-bordered-title{
  position: fixed;
  background-color: #efefef;
  color:$backcolor;
  padding-left: 20px;
  padding-right:20px;
  padding-top:5px;
  padding-bottom:5px;
  transform: translate3d(-10px,-10px,0px);
}
.alwen-left-menu-hr{
  background-image: linear-gradient(90deg, #7f7f7f,#7f7f7f,#7f7f7f, #222222);
  height:1px;
  margin-top:30px;
  margin-bottom:30px;
  width:105%;
}
.alwen-left-menu-choice-text{
  text-align:right;
  white-space:nowrap;
}
.alwen-left-menu-choice{
  text-align:right;
  color: rgb(154,154,154);
  transition: all 0.3s;
}
.alwen-left-menu-choice:hover{
  text-align:right;
  transform: translateX(30px);
  
  transition: all 0.3s;
}
.alwen-left-menu-choice:hover .alwen-left-menu-choice-text{
  color: #ffffff;
  text-shadow: 0.5px 0.5px 1px #cccccc;
  transition: all 0.3s;
}
.alwen-left-menu-choice-sr{
  display:relative;
  z-index: 99;
  color: $backcolor;
  transition: all 0.3s;
}
.alwen-left-menu-choice:hover .alwen-left-menu-choice-sr{
  color: #ffffff;
  text-shadow: 0.5px 0.5px 1px #cccccc;
  transition: all 0.3s;
}
.alwen-left-menu{
  top: 50%;
  transform: translate(0%,-50%);
  width:40%;
  position:fixed;
  left:0px;
  font-family: 'Geometos','SourceHanSerifHeavy',sans-serif;
  margin-top:15px;
  font-weight:normal;
}
//PC
@media screen and (min-width: 1200px) {
  .alwen-left-menu{
    font-size:24px;
  }
}
@media screen and (max-width: 1200px) {
  .alwen-left-menu{
    font-size:22px;
  }
}
.alwen-proton-right-sb{
  width:45%;
  position:fixed;
  right:0px;
  font-family: 'Geometos','SourceHan',sans-serif;
  margin-top:15px;
  font-size:13px;
  font-weight:normal;
}
.alwen_ak_hr{
  border-top: 1px dashed #efefef;
  width:45%;
  position:fixed;
  right:0px;
}
.alwen-proton-wgl{
  position:absolute;
  top: 40%;
  right:5%;
  transform: translate(0%,-50%);
}
.alwen-proton-title{
  text-align:right;
  position:absolute;
  top: 60%;
  right:10%;
  transform: translate(0%,-50%);
  font-family: 'Geometos','SourceHanSerifHeavy',sans-serif;
  font-size: 20px;
  width:100%;
}
.alwen-proton-title-v{
  font-size:36px;
}
#tsparticles{}
</style>
