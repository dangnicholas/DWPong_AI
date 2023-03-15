import http.server
import os
import socketserver

if __name__ == "__main__":
    PORT = 8000
    root_dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DIRECTORY = os.path.join(root_dirname, 'visualizer')
    print(os.path.abspath(DIRECTORY))


    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=DIRECTORY, **kwargs)


    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("serving at port", PORT)
        httpd.serve_forever()