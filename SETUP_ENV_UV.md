# uv를 사용하여 프로젝트 설정하기

이 저장소에는 `uv` 패키지 매니저를 사용하여 환경을 빠르고 효율적으로 설정하는 방법이 안내되어 있습니다. `uv`는 매우 빠른 Python 패키지 설치 및 관리 도구입니다.

## 1. uv 설치
`uv`가 설치되어 있지 않다면 아래 명령어로 설치할 수 있습니다.
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. 프로젝트 설정 및 동기화
프로젝트 폴더에서 다음 명령어를 실행하면 가상 환경 생성과 패키지 설치가 한 번에 완료됩니다.
```bash
uv sync
```
> [!NOTE]
> `.venv` 폴더가 없다면 자동으로 생성하고, `pyproject.toml` 및 `uv.lock`에 정의된 모든 의존성을 동기화합니다.

## 3. 코드 실행 (uv run)
가상 환경을 **활성화할 필요 없이** `uv run`을 사용하여 코드를 실행할 수 있습니다.
```bash
# 파이썬 스크립트 실행
uv run python main.py

# Jupyter Notebook 실행 (또는 IDE에서 커널 선택)
uv run jupyter notebook
```

## 4. 커널 선택 (VS Code 및 IDE)
Jupyter Notebook(`.ipynb`)을 사용하신다면 IDE 우측 상단의 **Kernel Select**에서 생성된 `.venv` 내의 Python 인터프리터를 선택해 주세요.

## 5. 패키지 관리
- **패키지 추가**: `uv add <package_name>`
- **패키지 제거**: `uv remove <package_name>`
- **의존성 업데이트**: `uv lock --upgrade`

---

### (참고) 수동으로 가상 환경 활성화하기
기존 방식처럼 `python` 명령어를 직접 사용하고 싶다면 가상 환경을 수동으로 활성화할 수 있습니다.
- **macOS / Linux**: `source .venv/bin/activate`
- **Windows**: `.venv\Scripts\activate`

---
> [!TIP]
> `uv run`은 명령어를 실행하기 전에 의존성이 최신 상태인지 자동으로 확인하므로 가장 안전하고 편리한 실행 방법입니다.
