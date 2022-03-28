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

export {
  registerUser,
  deleteUser,
  loginUser,
  getTime,
  modifyTime,
  getReports,
  insertReports,
};
