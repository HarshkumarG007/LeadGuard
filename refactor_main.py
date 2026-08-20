import glob

files = glob.glob("src/leadguard/**/*.py", recursive=True)
for file in files:
    with open(file, encoding="utf-8") as f:
        content = f.read()

    if 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content:
        lines = content.split("\n")
        new_lines = []
        in_main = False
        for line in lines:
            if line.startswith('if __name__ == "__main__":') or line.startswith(
                "if __name__ == '__main__':"
            ):
                in_main = True
                new_lines.append("def main():")
            elif in_main:
                if line.strip() == "":
                    new_lines.append(line)
                elif line.startswith(" " * 4) or line.startswith("\t"):
                    new_lines.append(line)
                else:
                    in_main = False
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if "def main():" in new_lines:
            # Add the call at the end
            new_lines.append("")
            new_lines.append('if __name__ == "__main__":')
            new_lines.append("    main()")

            with open(file, "w", encoding="utf-8") as f:
                f.write("\n".join(new_lines))
            print(f"Refactored {file}")
