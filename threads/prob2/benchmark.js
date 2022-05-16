const { execSync } = require("child_process");
const { writeFileSync } = require("fs");

for (let i = 1; i <= 10; i++) {
  let thread_times = [];

  for (let j = 0; j < 5; j++) {
    const process = execSync(
      `./a.out -i mat128_32.bin -i mat128_64.bin -i mat128_128.bin -i mat128_256.bin -t ${i}`
    );

    var out = process.toString().split("\n");
    var time = parseFloat(out[out.length - 2].split("=")[1].split(" ")[1]);
    thread_times.push(time);
  }

  writeFileSync(`${i}.txt`, thread_times.join(","));
  const sum = thread_times.reduce((a, b) => a + b, 0);
  const avg = sum / thread_times.length;
  console.log(`${i} Threads -> ${avg} s`);
}
