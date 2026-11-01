// 生成HTML内の推論関数を実行し、Goで記録した全サンプルと照合する。
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const html = fs.readFileSync(process.argv[2] || 'report-digits.html', 'utf8');
const match = html.match(/<script id="report-data" type="application\/json">([\s\S]*?)<\/script>/);
assert.ok(match, '実行データが見つかりません');
const data = JSON.parse(match[1]);
const start = html.indexOf('function infer('), end = html.indexOf('const pad=', start);
assert.ok(start >= 0 && end > start, '推論関数が見つかりません');
const context = vm.createContext({});
vm.runInContext(html.slice(start, end), context, { timeout: 1000 });
let maxError = 0;
const examples = [...data.samples, ...(data.mistakes || [])];
for (const sample of examples) {
  const pixels = Float64Array.from(Buffer.from(sample.pixels, 'base64'), value => value / 255);
  assert.equal(pixels.length, 784);
  const scores = context.infer(data.model, pixels);
  for (let i = 0; i < 10; i++) maxError = Math.max(maxError, Math.abs(scores[i] - sample.scores[i]));
  assert.equal(scores.indexOf(Math.max(...scores)), sample.prediction);
}
assert.ok(maxError < 1e-12, `スコアが一致しません: ${maxError}`);
assert.equal(data.confusion.flat().reduce((sum, value) => sum + value, 0), data.testCount);
const correct = data.confusion.reduce((sum, row, i) => sum + row[i], 0);
assert.equal(correct / data.testCount, data.test.accuracy);
assert.equal(data.history[0].epoch, 0);
console.log(`PASS: Go/JavaScript ${examples.length}枚一致、最大スコア差 ${maxError}、テスト正解 ${correct}/${data.testCount}`);
