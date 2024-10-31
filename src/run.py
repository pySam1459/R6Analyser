import cli

from utils.constants import RED, WHITE

if __name__ == "__main__":
    cli.main()


if __name__ == "__main__2":
    try:
        cli.main()
    except Exception as e:
        print(f"{RED}An Error Occurred!{WHITE}\n{e}")
