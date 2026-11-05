import subprocess
import sys
import time

if __name__ == "__main__":
    print("🚀 正在启动后端 FastAPI...")
    backend = subprocess.Popen([sys.executable, "backend/main.py"])

    # 给后端 3 秒钟启动时间
    time.sleep(3)

    print("\n🌐 正在启动前端 Gradio...")
    frontend = subprocess.Popen([sys.executable, "web_gradio.py"])

    print("\n✅ 启动完成！如果是第一次打开，请去浏览器访问: http://127.0.0.1:7860")
    print("按 Ctrl+C 可以关闭所有服务。\n")

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print("\n🛑 正在关闭服务...")
        backend.terminate()
        frontend.terminate()
