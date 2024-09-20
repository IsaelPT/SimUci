import subprocess


def run_streamlit_app() -> None:
    args = ["streamlit", "run", "streamlit/app.py"]
    subprocess.run(args)


if __name__ == "__main__":
    run_streamlit_app()
