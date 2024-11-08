<script setup>
import {ref, computed, reactive} from 'vue';
import {message} from 'ant-design-vue';
import {detection, labels, resetLabels, trainsItems, uploadFile, uploadFiles} from "@/api/files.js";
import {InboxOutlined} from '@ant-design/icons-vue';

let activeKey = ref('2');
let fileList = ref([]);
let items = ref([]);
let trains = ref([]);
let activateItem = ref({})
let detectResults = ref([])
let value = ref('a')
let value1 = ref('a')
let value2 = ref('a')
let uploading = ref(false)
let detectLoading = ref(false)

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
      console.log(detectResults.value, 'detectResults.value')
    }
    if (items.value.length > 0) {
      activateItem.value = {item: items.value[0], index: 0}
      // console.log(activateItem.value, 'ppppppppp')
    }
    message.success(res.imageInfo);
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

function handlerCustomRequest() {
  const formData = new FormData();
  fileList.value.forEach(file => {
    formData.append('files[]', file.originFileObj);
  });

  uploading.value = true;

  uploadFiles(formData)
    .then(res => {
      console.log(res, 'uploadFiles')
      // fileList.value = [];
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
  console.log(radio)
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
  <main style="margin: 0 18px;padding-bottom: 18px">
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
              <a-row justify="start" :gutter="16">
                <a-col :span="6">
                  <div style="height: 22px">
                    <a-radio-group v-model:value="value" button-style="solid" size="small"
                                   @change="handlerChangeDatasets">
                      <a-radio-button value="a">结果集</a-radio-button>
                      <a-radio-button value="b">测试集</a-radio-button>
                    </a-radio-group>
                  </div>
                  <a-divider/>
                  <ul class="items" v-show="value === 'a'">
                    <li v-for="(item, index) in items" :key="index" class="item"
                        @click="handlerActivateItem(item, index)">
                      {{ item.filename }}
                    </li>
                  </ul>
                  <ul class="items" v-show="value === 'b'">
                    <li v-for="(item, index) in trains" :key="index" class="item"
                        @click="handlerActivateItem(item, index)">
                      {{ item.filename }}
                    </li>
                  </ul>
                </a-col>
                <a-col :span="10">
                  <div>预览</div>
                  <a-divider/>
                  <div class="image-wrapper">
                    <a-image v-if="activateItem.item && activateItem.item.path" :src="activateItem.item.path"
                             width="100%"/>
                  </div>
                </a-col>
                <a-col :span="8">
                  <div>结果</div>
                  <a-divider/>
                  <div style="display: none">{{ activateDetectResult }}</div>
                  <ul class="items">
                    <li v-for="(item, index) in cls" :key="index" class="item">
                      <div>类别：<span style="font-weight: 600">{{ category[index] }}</span></div>
                      <div>概率：<span style="font-weight: 600">{{ conf[index] }}</span></div>
                      <div>坐标：<span style="font-weight: 600">{{
                          xywh[index].map(item => Number(item).toFixed(2))
                        }}</span></div>
                    </li>
                  </ul>
                </a-col>
              </a-row>
            </a-card>
          </a-col>
        </a-row>
      </a-tab-pane>
      <a-tab-pane key="3" tab="视频检测">Content of Tab Pane 3</a-tab-pane>
    </a-tabs>
  </main>
</template>


<style scoped lang="scss">
ul {
  list-style: none;
  margin: 0;
  padding: 0;
}

.image-wrapper {
  height: 500px;
}

.items {
  height: 500px;
  overflow: auto;

  .item {
    padding: 8px 16px 8px 0;
    border-block-end: 1px solid rgba(5, 5, 5, 0.06);
  }
}

::v-deep .ant-upload-list-text {
  max-height: 200px;
  overflow: auto;
}
</style>
