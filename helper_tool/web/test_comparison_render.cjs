// Interaction/rendering contracts with synthetic aggregates; no browser/data files.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const elements=new Map();
function element(id){
  if(!elements.has(id))elements.set(id,{id,innerHTML:'',textContent:'',hidden:false,value:'',options:[],listeners:{},
    classList:{toggle(){},add(){},remove(){}},addEventListener(name,fn){this.listeners[name]=fn;},
    querySelectorAll(){return[];},querySelector(){return element('option');}});
  return elements.get(id);
}
const context=vm.createContext({document:{getElementById:element,querySelectorAll:()=>[],querySelector:()=>element('container')},
  location:{hash:''},history:{replaceState(){}},setTimeout:()=>1,clearTimeout(){},console,
  fetch:async()=>({ok:true,json:async()=>({devices:[],sources:[],experiments:[],skipped:0})})});
vm.runInContext(fs.readFileSync(require('node:path').join(__dirname,'app.js'),'utf8'),context);
(async()=>{
  await new Promise(setImmediate);
  context.rows=[{run_id:'20261115-080001-aaaaaaaa',label:'<script>synthetic</script>',dataset:'synthetic.jsonl',count:2,
    dataset_sha256:'a'.repeat(64),method:'bge-lr',limit:'native',model_path:'/models/run-one',total:.712345,
    metrics:{category:.612345,device:.8,urgency:.9,duplicate:.7},timing:{total_seconds:2.1234,prediction_seconds:1.1234}},
    {run_id:'20261115-080000-bbbbbbbb',label:'second',dataset:'synthetic.jsonl',count:2,dataset_sha256:'a'.repeat(64),
    method:'bge-svc',limit:'2gb',model_path:'/models/run-two',total:null,
    metrics:{category:null,device:null,urgency:null,duplicate:null},timing:{total_seconds:3,prediction_seconds:null}}];
  vm.runInContext('comparisonExperiments=rows;renderExperimentComparison()',context);
  assert.equal(element('comparison-results').hidden,true);
  element('comparison-search').value='SVC';element('comparison-select').listeners.click();
  assert.ok(element('comparison-count').textContent.includes('已選 1 筆'));
  assert.ok(element('comparison-rows').innerHTML.includes('未評分'));
  assert.ok(!element('comparison-rows').innerHTML.includes('0.000000'));
  element('comparison-search').value='';element('comparison-select').listeners.click();
  assert.ok(element('comparison-count').textContent.includes('已選 2 筆'));
  assert.equal(element('comparison-results').hidden,false);
  assert.ok(element('comparison-rows').innerHTML.includes('0.712345'));
  assert.ok(element('comparison-rows').innerHTML.includes('1.123'));
  assert.ok(element('comparison-rows').innerHTML.includes('&lt;script&gt;'));
  assert.ok(!element('comparison-rows').innerHTML.includes('<script>'));
  assert.ok(element('comparison-warning').textContent.includes('相同資料'));
  vm.runInContext("comparisonExperiments[1].dataset_sha256='b'.repeat(64);renderExperimentComparison()",context);
  assert.ok(element('comparison-warning').textContent.includes('不同資料'));
  element('comparison-clear').listeners.click();
  assert.equal(element('comparison-results').hidden,true);
  assert.ok(element('comparison-count').textContent.includes('已選 0 筆'));
  console.log('Comparison rendering: selection, filtering, clearing, scores, missing values, escaping and data identity passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
