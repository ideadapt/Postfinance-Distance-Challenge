const post = {id: 0, x: 3, y: 5};

const a1 = {id: 1, x: 10, y: 8};
const a2 = {id: 2, x: 11, y: 3};
const a3 = {id: 3, x: 14, y: 7};
const a4 = {id: 4, x: 15, y: 1};
const a5 = {id: 5, x: 18, y: 5};
const power = {id: 6, x: 22, y: 2};

const a11 = {id: 11, x: 26, y: 4};
const a12 = {id: 12, x: 35, y: 8};
const a13 = {id: 13, x: 26, y: 9};
const a14 = {id: 14, x: 28, y: 13};
const a15 = {id: 15, x: 19, y: 9};
const edu = {id: 16, x: 19, y: 13};

const a21 = {id: 21, x: 21, y: 15};
const a22 = {id: 22, x: 24, y: 17};
const a23 = {id: 23, x: 30, y: 15};
const a24 = {id: 24, x: 34, y: 15};
const a25 = {id: 25, x: 46, y: 15};
const dsd = {id: 26, x: 38, y: 14};

const a31 = {id: 31, x: 42, y: 13};
const a32 = {id: 32, x: 41, y: 2};
const a33 = {id: 33, x: 46, y: 9};
const a34 = {id: 34, x: 48, y: 3};
const a35 = {id: 35, x: 51, y: 14};
const char ={id: 36, x: 54, y: 8};

const neighbours = new Map();
neighbours.set(post, new Set([a1, a2]));
neighbours.set(a1, new Set([a3, post]));
neighbours.set(a2, new Set([post, a4, a5]));
neighbours.set(a3, new Set([a5]));
neighbours.set(a4, new Set([power]));
neighbours.set(a5, new Set([a2, a3, power]));
neighbours.set(power, new Set([a13, a11, a4, a5]));

neighbours.set(a11, new Set([power, a12]));
neighbours.set(a12, new Set([a13, a14]));
neighbours.set(a13, new Set([power, a12, a14, a15]));
neighbours.set(a14, new Set([a12, a13, edu]));
neighbours.set(a15, new Set([a13, edu]));
neighbours.set(edu, new Set([a15, a14, a21]));

neighbours.set(a21, new Set([edu, a22]));
neighbours.set(a22, new Set([a23, a24]));
neighbours.set(a23, new Set([a22, dsd]));
neighbours.set(a24, new Set([a22, a25]));
neighbours.set(a25, new Set([a24, dsd]));
neighbours.set(dsd, new Set([a25, a31]));

neighbours.set(a31, new Set([dsd, a32, a35]));
neighbours.set(a32, new Set([a31, a33, a34]));
neighbours.set(a33, new Set([a32, char]));
neighbours.set(a34, new Set([a32, char]));
neighbours.set(a35, new Set([a31, char]));
neighbours.set(char, new Set([a34, a35]));



const dist = (a, b) => Math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2)

/**
 for each node id (0),
 store its related node ids (1, 2)
 and their distances (6, 7)
 { 0: { 1: 6, 2: 7}, 1: ... }
 */
const problem = {}
for(const [node, nbs] of neighbours.entries()){
  problem[node.id] = {}
  for(const neighbour of nbs.values()){
    problem[node.id][neighbour.id] = dist(node, neighbour)
  }
}

const lowestCostNode = (costs, processed) => {
  return Object.keys(costs).reduce((lowest, node) => {
    if (lowest === null || costs[node] < costs[lowest]) {
      if (!processed.includes(node)) {
        lowest = node;
      }
    }
    return lowest;
  }, null);
};

// based on https://gist.github.com/MoeweX/ab98efee9435b47529e3a6cb50c5b605
const dijkstra = (graph, startNodeName, endNodeName) => {
  // track the lowest cost to reach each node
  let costs = {};
  costs[endNodeName] = "Infinity";
  Object.assign(costs, graph[startNodeName]);

  // track paths
  const parents = {endNodeName: null};
  for (const child of Object.keys(graph[startNodeName])) {
    parents[child] = startNodeName;
  }

  // track nodes that have already been processed
  const processed = [];

  let node = lowestCostNode(costs, processed);
  while (node !== null) {
    const cost = costs[node]
    const adjs = graph[node]
    for (const adj of Object.keys(adjs)) {
      if (adj !== startNodeName) {
        const newCost = cost + adjs[adj]
        if (costs[adj] == null || costs[adj] > newCost) {
          costs[adj] = newCost
          parents[adj] = node
        }
      }
    }
    processed.push(node);
    node = lowestCostNode(costs, processed);
  }

  let optimalPath = [endNodeName]
  let parent = parents[endNodeName];
  while (parent) {
    optimalPath.push(parent);
    parent = parents[parent];
  }
  optimalPath.push(startNodeName);
  optimalPath = optimalPath.reverse().map(Number);

  return {
    distance: costs[endNodeName],
    path: optimalPath
  };
};



const data = dijkstra(problem, post.id, char.id)
console.log(data.distance, data.path);



// visualize it using P5.js, for fun
function setup() {
  createCanvas(1200, 400)
  background(220)

  textSize(22)
  fill('red')
  const f = 20
  for(const [node, ns] of neighbours.entries()){
    const [nx, ny] = [node.x*f, node.y*f]
    circle(nx, ny, 5)
    fill('blue')
    text(node.id, nx+5, ny+5)
    fill('red')

    for(const neighbour of ns.values()){
      const [nbx, nby] = [neighbour.x*f, neighbour.y*f]
      circle(nbx, nby, 5)
      line(nx, ny, nbx, nby)
      const distance = Number(dist({ x: nx/f, y: ny/f }, { x: nbx/f, y: nby/f })).toFixed(1);
      textSize(16)
      text(distance, (nbx+nx)/2, (ny+nby)/2)
      textSize(22)
      fill('blue')
      text(neighbour.id, nbx+5, nby+5)
      fill('red')
    }
  }

  const byId = new Map(Array.from(neighbours.entries()).map(([a]) => [a.id, a]))
  let prevId = null
  for(const aId of data.path){
    if(prevId !== null){
      stroke('orange')
      const prevNode = byId.get(prevId)
      const node = byId.get(aId)
      line(prevNode.x*f, prevNode.y*f, node.x*f, node.y*f)
    }
    prevId = aId
  }
}