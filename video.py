import sys

def main(argv):
  if len(argv) != 3:
    print("Expected exactly 2 arguments: <model-id> <video.mp4>")

if __name__ == "__main__":
  main(sys.argv)