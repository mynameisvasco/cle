const { execSync } = require("child_process");

for (let i = 1; i <= 8; i++) {
  let thread_times = [];

  for (let j = 0; j < 10; j++) {
    const process = execSync(
      `./a.out -i text0.txt -i text1.txt -i text2.txt -i text3.txt -i text4.txt -t ${i}`
    );

    var out = process.toString().split("\n");
    var time = parseFloat(out[out.length - 2].split("=")[1].split(" ")[1]);
    thread_times.push(time);
  }

  const sum = thread_times.reduce((a, b) => a + b, 0);
  const avg = sum / thread_times.length;
  console.log(`${i} Threads -> ${avg} s`);
}
