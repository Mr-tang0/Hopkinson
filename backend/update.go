/*
 * @Author: tang
 * @Date: 2026-05-23
 * @GitHub: Mr-tang0/CTSystem
 * @Description: 更新服务模块，负责从GitHub获取版本更新信息
 */
package backend

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

//此文件为获取更新信息的接口

type UpdateService struct {
	cachedRelease *GitHubRelease
}

type GitHubRelease struct {
	TagName string `json:"tag_name"`
	HTMLURL string `json:"html_url"`
	Assets  []struct {
		Name               string `json:"name"`
		BrowserDownloadURL string `json:"browser_download_url"`
	} `json:"assets"`
}

type UpdateResult struct {
	CurrentVersion string `json:"currentVersion"`
	LatestVersion  string `json:"latestVersion"`
	DownloadURL    string `json:"downloadUrl"`
}

func (u *UpdateService) GetUpdateInfo() (UpdateResult, error) {
	path := "update.json"
	appJson, err := os.ReadFile(path)
	if err != nil {
		fmt.Printf("读取文件失败: %v\n", err)
		return UpdateResult{}, err
	}

	var appInfo struct {
		Name    string `json:"name"`
		Version string `json:"version"`
		URL     string `json:"url"`
	}
	err = json.Unmarshal(appJson, &appInfo)
	if err != nil {
		return UpdateResult{}, err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(appInfo.URL)
	if err != nil {
		fmt.Printf("HTTP请求失败: %v\n", err)
		return UpdateResult{}, err
	}
	defer resp.Body.Close()

	fmt.Printf("GitHub 返回状态码: %d\n", resp.StatusCode)
	if resp.StatusCode != 200 {
		return UpdateResult{}, errors.New("获取最新发布信息失败")
	}

	bodyBytes, _ := io.ReadAll(resp.Body)

	var release GitHubRelease
	if err := json.Unmarshal(bodyBytes, &release); err != nil {
		fmt.Printf("JSON 解析失败: %v\n", err)
		return UpdateResult{}, err
	}

	u.cachedRelease = &release

	println("当前版本:", appInfo.Version)
	println("最新版本:", release.TagName)

	if release.TagName == appInfo.Version {
		return UpdateResult{}, errors.New("当前版本已经是最新版")
	}

	downloadUrl := release.HTMLURL
	if len(release.Assets) > 0 {
		downloadUrl = release.Assets[0].BrowserDownloadURL
	}

	return UpdateResult{
		CurrentVersion: appInfo.Version,
		LatestVersion:  release.TagName,
		DownloadURL:    downloadUrl,
	}, nil
}

func (u *UpdateService) GetCachedRelease() GitHubRelease {
	if u.cachedRelease == nil {
		fmt.Println("release info not available")
		return GitHubRelease{}
	}
	fmt.Println("release info available")
	return *u.cachedRelease
}
