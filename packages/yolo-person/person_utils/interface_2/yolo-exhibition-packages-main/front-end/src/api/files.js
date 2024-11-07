import request from "@/utils/request.js";

export function uploadFile(file) {
  return request({
    url: '/uploadFile',
    method: 'post',
    data: file,
    onUploadProgress: function (progressEvent) {
      let {total, loaded} = progressEvent
      console.log(loaded, 'loaded')
    },
  })
}

export function uploadFiles(files) {
  return request({
    url: '/uploadFiles',
    method: 'post',
    data: files,
    onUploadProgress: function (progressEvent) {
      let {total, loaded} = progressEvent
      console.log(loaded, 'loaded')
    },
  })
}

export function detection() {
  return request({
    url: '/detection',
    method: 'get'
  })
}

export function videoDetection(name) {
  return request({
    url: '/videoDetection/' + name,
    method: 'get'
  })
}

export function videoLabels(name) {
  return request({
    url: '/videos/' + name,
    method: 'get'
  })
}

export function labels() {
  return request({
    url: '/labels',
    method: 'get'
  })
}

export function trainsItems() {
  return request({
    url: '/trains',
    method: 'get'
  })
}

export function detectionLabel(label) {
  return request({
    url: '/detection/' + label,
    method: 'get'
  })
}

export function resetLabels() {
  return request({
    url: '/reset',
    method: 'get',
  })
}
