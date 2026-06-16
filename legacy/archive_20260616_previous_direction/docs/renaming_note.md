# Renaming Note

- 旧名称: `MockVLM`
- 新名称: `HomeServiceActionVerifier`
- 旧Python package: `privacy_vlm_poc`
- 新Python package: `home_service_action_verifier`
- 旧CLI: `privacy-vlm-poc`
- 新CLI: `home-service-verifier`

GitHub上のリポジトリ名変更は、Codexから直接設定変更できない可能性があります。必要に応じて GitHub の repository settings で `DaichiHiraoka/MockVLM` から `DaichiHiraoka/HomeServiceActionVerifier` へ手動変更してください。

リモートURL変更後、ローカルでは次を実行します。

```powershell
git remote -v
git remote set-url origin https://github.com/DaichiHiraoka/HomeServiceActionVerifier.git
```
