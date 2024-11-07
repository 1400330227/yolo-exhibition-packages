import axios from 'axios'

const service = axios.create({baseURL: import.meta.env.VITE_BASE_API, timeout: 5000000})

service.interceptors.request.use()
service.interceptors.response.use(
  response => {
    const res = response.data
    const status = response.status
    if (status !== 200) {
      if (res.code === 50008 || res.code === 50012 || res.code === 50014) {
      }
      return Promise.reject(new Error(res.message || 'Error'))
    } else {
      return res
    }
  },
  error => {
    console.log('err' + error) // for debug
    return Promise.reject(error)
  }
)

export default service

