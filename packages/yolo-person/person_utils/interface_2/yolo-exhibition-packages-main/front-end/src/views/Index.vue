<script setup>
import {ref, computed, reactive} from 'vue';
import {message} from 'ant-design-vue';
import {detection, labels, resetLabels, trainsItems, videoLabels, uploadFiles, videoDetection} from "@/api/files.js";
import {InboxOutlined} from '@ant-design/icons-vue';
import 'video.js/dist/video-js.css'

let activeKey = ref('2');
let fileList = ref([]);
let videoList = ref([]);
let items = ref([]);
let trains = ref([]);
let activateItem = ref({})
let detectResults = ref([])
let value = ref('a')
let value1 = ref('a')
let value2 = ref('a')
let uploading = ref(false)
let detectLoading = ref(false)
let videoDetectLoading = ref(false)
let videoUrl = ref('')
let filterValue = ref('')

let activateDetectResult = computed(() => {
  let result = {}
  let {index} = activateItem.value;
  if (index != null && detectResults.value[index] != null) {
    result = detectResults.value[index]
    if (Array.isArray(result['conf'])) {
      result['conf'] = result['conf'].map(item => (Number(item)).toFixed(2))
    }
  }
  return result;
});

let cls = computed(() => activateDetectResult.value['cls'])
let conf = computed(() => activateDetectResult.value['conf'])
let xywh = computed(() => activateDetectResult.value['xywh'])
let category = computed(() => activateDetectResult.value['category'])
let data = computed(() => activateDetectResult.value['data'])

/**
 * 目标检测
 * @param event
 */
async function handlerDetect(event) {
  detectLoading.value = true
  let res = await detection();
  if (res.status === 200) {
    detectLoading.value = false
    await handlerGetLabels();
    await handlerGetTrains();
    if (res.detectResults != null) {
      detectResults.value = JSON.parse(res.detectResults)
      // console.log(detectResults.value, 'detectResults.value')
    }
    if (items.value.length > 0) {
      activateItem.value = {item: items.value[0], index: 0}
    }
    message.success(res.imageInfo);
  } else if (res.status === 400) {
    message.info('请上传文件')
    detectLoading.value = false
  }
}

async function handlerVideoDetect(event) {
  videoDetectLoading.value = true;
  console.log(videoList.value, 'videoList')
  let name = ''
  if (videoList.value.length > 0) {
    name = videoList.value[0].name
  } else {
    return message.info('请上传视频')
  }
  let res = await videoDetection(name);
  if (res.status === 200) {

    let url = res.videoUrl
    if (url) {
      videoUrl.value = '/api/videos/' + url;
    }
    videoDetectLoading.value = false
    message.success(res.imageInfo);
  } else {
    videoDetectLoading.value = false
    message.error('检测失败')
  }
}

function handlerActivateItem(item, index) {
  activateItem.value = {item, index}
}

function handlerResetLabels() {
  resetLabels().then(() => {
    message.success('重置成功！');
    location.reload()
  })
}

async function handlerGetLabels() {
  let result = await labels();
  items.value = result.map(item => {
    return {filename: item, path: '/api/detection/' + item}
  });
}

async function handlerGetTrains() {
  let result = await trainsItems();
  trains.value = result.map(item => {
    return {filename: item, path: '/api/training/' + item}
  });
}

function handlerBeforeUpload(file) {
  fileList.value = [...(fileList.value || []), file];
  return false;
}

function handlerBeforeUploadVideo(file) {
  videoList.value = [...(videoList.value || []), file];
  return false;
}

function handlerCustomRequest() {
  const formData = new FormData();
  fileList.value.forEach(file => {
    formData.append('files[]', file.originFileObj);
  });

  uploading.value = true;

  uploadFiles(formData)
    .then(res => {
      console.log(res, 'uploadFiles')
      uploading.value = false;
      message.success('上传成功');
    })
    .then(() => {
      handlerGetTrains()
    })
    .catch(err => {
      uploading.value = false;
      message.error('upload failed.');
    })
}

function handlerVideoCustomRequest() {
  const formData = new FormData();
  videoList.value.forEach(file => {
    formData.append('files[]', file.originFileObj);
  });

  uploading.value = true;

  uploadFiles(formData)
    .then(res => {
      console.log(res, 'uploadFiles')
      uploading.value = false;
      message.success('上传成功');
    })
    .then(() => {
      handlerGetTrains()
    })
    .catch(err => {
      uploading.value = false;
      message.error('upload failed.');
    })
}


function handlerChangeDatasets(radio) {
  let value = radio.target.value;
}

const handleChange = info => {
  const {status, percent} = info.file;
  if (status === 'done') {
  }
}
const handleDrop = () => {
}
</script>

<template>
  <!-- margin: 0 18px;padding-bottom: 18px -->
  <main class="wrapper">
    <div class="wrapper__content">
      <a-tabs v-model:activeKey="activeKey">
        <a-tab-pane key="2" tab="图片检测" force-render>
          <a-row justify="center">
            <a-col :span="20">
              <a-card title="上传图片">
                <template #extra>
                  <a-button type="primary" @click="handlerCustomRequest" :loading="uploading"
                            :disabled="fileList.length === 0">上传文件
                  </a-button>
                </template>
                <a-upload-dragger
                  :before-upload="handlerBeforeUpload"
                  v-model:fileList="fileList"
                  name="file"
                  :multiple="true"
                  action="/api/uploadFile"
                  @change="handleChange"
                  @drop="handleDrop"
                  :max-count="50"
                >
                  <p class="ant-upload-drag-icon">
                    <InboxOutlined/>
                  </p>
                  <p class="ant-upload-text">选择或者拖拽文件上传</p>
                  <p class="ant-upload-hint">
                    最多上传50个文件，支持pdf、 png、 jpg、 jpeg、 gif格式
                  </p>
                </a-upload-dragger>
              </a-card>

            </a-col>
          </a-row>
          <div style="height: 12px"></div>
          <a-row justify="center">
            <a-col :span="20">
              <a-card title="检测信息">
                <template #extra>
                  <a-space wrap>
                    <a-popconfirm placement="topLeft" ok-text="确定" cancel-text="取消" @confirm="handlerResetLabels">
                      <template #title>
                        <p>确认重置检测信息？</p>
                      </template>
                      <a-button>重 置</a-button>
                    </a-popconfirm>
                    <a-button type="primary" @click="handlerDetect" :loading="detectLoading">开始检测</a-button>
                  </a-space>
                </template>
                <a-row justify="start" :gutter="0">
                  <a-col :span="12">
                    <div>测试集</div>
                    <a-divider/>
                  </a-col>
                  <a-col :span="12">
                    <div>结果集</div>
                    <a-divider/>
                  </a-col>
                </a-row>
                <a-row justify="start" :gutter="16" style="">
                  <a-col :span="12">
                    <ul class="items">
                      <li v-for="(item, index) in trains" :key="index" class="item">
                        <a-image :src="item.path" width="100%"></a-image>
                        <div style="text-align: center">{{ item.filename }}</div>
                      </li>
                    </ul>
                  </a-col>
                  <a-col :span="12">
                    <ul class="items">
                      <li v-for="(item, index) in items" :key="index" class="item">
                        <a-image :src="item.path" width="100%"></a-image>
                        <div style="text-align: center">{{ item.filename }}</div>
                      </li>
                    </ul>
                  </a-col>
                </a-row>
              </a-card>
            </a-col>
          </a-row>


          <div style="height: 12px"></div>
          <a-row justify="center">
            <a-col :span="20">
              <a-card title="信息详情">
                <template #extra>
                  <a-input v-model:value="filterValue" placeholder="输入名称搜索"/>
                </template>

                <ul class="category-items">
                  <li v-for="(item, index) in items" :key="index" class="item">
                    <div>名称：<span style="font-weight: 600">{{ item.filename }}</span></div>
                    <div>类别：<span
                      style="font-weight: 600">{{
                        detectResults[index] && detectResults[index]['category'] || ''
                      }}</span>
                    </div>
                    <div>概率：<span style="font-weight: 600">{{
                        detectResults[index] && detectResults[index]['conf'] &&
                        detectResults[index]['conf'].map(item => Number(item).toFixed(2)) || ''
                      }}</span></div>
                    <div>
                      <div style="float: left">坐标：</div>
                      <div style="margin-left: 42px;">
                        <template
                          v-if="detectResults[index] && detectResults[index]['xywh'] && Array.isArray(detectResults[index]['xywh'])">
                          <div style="" v-for="xywhs in detectResults[index]['xywh']">
                            <div style="display: block;font-weight: 600">{{
                                xywhs.map(inner => Number(inner).toFixed(1))
                              }}
                            </div>
                          </div>
                        </template>
                      </div>
                    </div>
                  </li>
                </ul>
              </a-card>
            </a-col>
          </a-row>
        </a-tab-pane>
        <a-tab-pane key="3" tab="视频检测">
          <a-row justify="center">
            <a-col :span="20">
              <a-card title="上传视频">
                <template #extra>
                  <a-button type="primary" @click="handlerVideoCustomRequest" :loading="uploading"
                            :disabled="videoList.length === 0">上传文件
                  </a-button>
                </template>
                <a-upload-dragger
                  :before-upload="handlerBeforeUploadVideo"
                  v-model:fileList="videoList"
                  name="file"
                  :multiple="true"
                  action="/api/uploadFile"
                  @change="handleChange"
                  @drop="handleDrop"
                  :max-count="1"
                >
                  <p class="ant-upload-drag-icon">
                    <InboxOutlined/>
                  </p>
                  <p class="ant-upload-text">选择或者拖拽文件上传</p>
                  <p class="ant-upload-hint">
                    最多上传1个文件，支持mp4格式
                  </p>
                </a-upload-dragger>
              </a-card>
            </a-col>
          </a-row>

          <div style="height: 12px"></div>
          <a-row justify="center">
            <a-col :span="20">
              <a-card title="检测信息">
                <template #extra>
                  <a-space wrap>
                    <a-popconfirm placement="topLeft" ok-text="确定" cancel-text="取消" @confirm="handlerResetLabels">
                      <template #title>
                        <p>确认重置检测信息？</p>
                      </template>
                      <a-button>重 置</a-button>
                    </a-popconfirm>
                    <a-button type="primary" @click="handlerVideoDetect" :loading="videoDetectLoading">开始检测
                    </a-button>
                  </a-space>
                </template>

                <div style="text-align: center">
                  <video v-if="videoUrl" height="600" controls>
                    <source :src="videoUrl" type="video/webm">
                    Your browser does not support the video tag.
                  </video>
                </div>
              </a-card>
            </a-col>
          </a-row>
        </a-tab-pane>
      </a-tabs>
    </div>
  </main>
</template>


<style scoped lang="scss">
.wrapper {
  //background: #F8F9FB;
  background: #f8f8ff;
  min-height: 100vh;

  padding-bottom: 24px;

  &__content {

    ::v-deep .ant-tabs-nav-wrap {
      background: #ffffff;
      padding: 0 18px;
    }
  }
}

ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

.image-wrapper {
  height: 500px;
}

.items {
  //height: 500px;
  //overflow: auto;
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-row-gap: 3px;
  grid-column-gap: 3px;

  .item {

    img {

    }

    //padding: 8px 16px 8px 0;
    //border-block-end: 1px solid rgba(5, 5, 5, 0.06);
  }
}

.category-items {
  display: grid;
  grid-template-columns: repeat(2, 1fr);

  .item {
    border-block-end: 1px solid rgba(5, 5, 5, 0.06);
  }
}

::v-deep .ant-upload-list-text {
  max-height: 200px;
  overflow: auto;
}
</style>
