import axios, { AxiosInstance } from "axios";

const API_BASE_URL = "http://j6b201.p.ssafy.io:8000/api";

function apiInstance(): AxiosInstance {
  const instance = axios.create({
    baseURL: API_BASE_URL,
    headers: {
      "Content-type": "application/json",
    },
  });
  return instance;
}

function loginApiInstance(): AxiosInstance {
  const jwtToken = sessionStorage.getItem("accessToken");
  const instance = axios.create({
    baseURL: API_BASE_URL,
    headers: {
      "Content-type": "application/json",
      Authorization: `Bearer ${jwtToken}`,
    },
  });
  return instance;
}

export { apiInstance, loginApiInstance };
