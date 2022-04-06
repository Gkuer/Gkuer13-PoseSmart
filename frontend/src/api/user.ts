import { AxiosInstance, AxiosResponse } from "axios";
import { apiInstance, loginApiInstance } from "./index";

const api: AxiosInstance = apiInstance();

interface User {
  username: string;
  password: string;
}
interface Time {
  blink_time: number;
  neck_time: number;
  stretching_time: number;
}

async function registerUser(user: User): Promise<void> {
  await api.post(`/accounts/`, JSON.stringify(user));
}

async function deleteUser(user: User): Promise<void> {
  const loginApi = loginApiInstance();
  await loginApi.delete(`/accounts/`);
}

async function loginUser(user: User): Promise<AxiosResponse<any, any>> {
  return await api.post(`/accounts/login/`, JSON.stringify(user));
}

async function getTime(): Promise<any> {
  const loginApi = loginApiInstance();
  return await loginApi.get(`/accounts/time/`);
}

async function modifyTime(time: Time): Promise<void> {
  const loginApi = loginApiInstance();
  await loginApi.put(`/accounts/time/`, JSON.stringify(time));
}

async function getReports(): Promise<any> {
  const loginApi = loginApiInstance();
  return await loginApi.get(`/reports/`);
}

async function insertReports(reports: any): Promise<any> {
  const loginApi = loginApiInstance();
  await loginApi.post(`/reports/`, JSON.stringify(reports));
}

async function getDetect(data: any) {
  const loginApi = loginApiInstance();
  const temp = data;
  if (temp.cnt <= 4) {
    temp.face_x = data.face_x.toString();
    temp.face_y = data.face_y.toString();
    temp.nose_to_center = data.nose_to_center.toString();
  }
  return await loginApi.post(`/detections/`, JSON.stringify(temp));
}

async function getDetectBlink(data: any) {
  const loginApi = loginApiInstance();
  return await loginApi.post(`/detections/blink/`, JSON.stringify(data));
}

export {
  registerUser,
  deleteUser,
  loginUser,
  getTime,
  modifyTime,
  getReports,
  insertReports,
  getDetect,
  getDetectBlink,
};
