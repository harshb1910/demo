const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");

const scoreEl = document.getElementById("score");
const finalScoreEl = document.getElementById("final-score");
const overlay = document.getElementById("overlay");
const gameOverOverlay = document.getElementById("gameover");
const startBtn = document.getElementById("start");
const restartBtn = document.getElementById("restart");

const groundY = canvas.height - 60;
const gravity = 0.7;
const lift = -12;

const player = {
  x: 120,
  y: groundY,
  width: 36,
  height: 36,
  velocityY: 0,
  isGrounded: true,
};

let obstacles = [];
let stars = [];
let score = 0;
let speed = 5;
let animationId = null;
let isRunning = false;
let spacePressed = false;

const resetPlayer = () => {
  player.y = groundY;
  player.velocityY = 0;
  player.isGrounded = true;
};

const resetGame = () => {
  obstacles = [];
  stars = createStars(80);
  score = 0;
  speed = 5;
  resetPlayer();
  updateScore();
};

const createStars = (count) =>
  Array.from({ length: count }, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    radius: Math.random() * 1.5 + 0.5,
    speed: Math.random() * 0.6 + 0.2,
  }));

const updateScore = () => {
  scoreEl.textContent = Math.floor(score).toString();
};

const spawnObstacle = () => {
  const height = Math.random() * 30 + 30;
  const width = Math.random() * 28 + 22;
  obstacles.push({
    x: canvas.width + width,
    y: groundY + player.height - height,
    width,
    height,
  });
};

const drawStars = () => {
  ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
  stars.forEach((star) => {
    ctx.beginPath();
    ctx.arc(star.x, star.y, star.radius, 0, Math.PI * 2);
    ctx.fill();
  });
};

const updateStars = () => {
  stars.forEach((star) => {
    star.x -= star.speed;
    if (star.x < 0) {
      star.x = canvas.width + Math.random() * 40;
      star.y = Math.random() * canvas.height;
    }
  });
};

const drawPlayer = () => {
  ctx.fillStyle = "#9aa8ff";
  ctx.fillRect(player.x, player.y, player.width, player.height);
  ctx.fillStyle = "#f7f5ff";
  ctx.fillRect(player.x + 6, player.y + 8, player.width - 12, player.height - 14);
  ctx.fillStyle = "#3940a8";
  ctx.fillRect(player.x + 12, player.y + 14, 6, 6);
  ctx.fillRect(player.x + 22, player.y + 14, 6, 6);
};

const drawObstacles = () => {
  ctx.fillStyle = "#ff7a7a";
  obstacles.forEach((obstacle) => {
    ctx.fillRect(obstacle.x, obstacle.y, obstacle.width, obstacle.height);
  });
};

const drawGround = () => {
  ctx.fillStyle = "#151937";
  ctx.fillRect(0, groundY + player.height, canvas.width, canvas.height - groundY);
  ctx.strokeStyle = "#3a4173";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, groundY + player.height);
  ctx.lineTo(canvas.width, groundY + player.height);
  ctx.stroke();
};

const updatePlayer = () => {
  if (!player.isGrounded) {
    player.velocityY += gravity;
    if (spacePressed && player.velocityY < 0) {
      player.velocityY += gravity * -0.4;
    }
  }

  player.y += player.velocityY;

  if (player.y >= groundY) {
    player.y = groundY;
    player.velocityY = 0;
    player.isGrounded = true;
  }
};

const updateObstacles = () => {
  obstacles.forEach((obstacle) => {
    obstacle.x -= speed;
  });

  if (obstacles.length === 0 || obstacles[obstacles.length - 1].x < canvas.width - 240) {
    if (Math.random() > 0.6) {
      spawnObstacle();
    }
  }

  obstacles = obstacles.filter((obstacle) => obstacle.x + obstacle.width > 0);
};

const checkCollision = () => {
  return obstacles.some((obstacle) =>
    player.x < obstacle.x + obstacle.width &&
    player.x + player.width > obstacle.x &&
    player.y < obstacle.y + obstacle.height &&
    player.y + player.height > obstacle.y
  );
};

const drawScene = () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  updateStars();
  drawStars();
  drawGround();
  drawPlayer();
  drawObstacles();
};

const update = () => {
  if (!isRunning) {
    return;
  }

  updatePlayer();
  updateObstacles();
  drawScene();

  score += 0.12;
  if (score % 20 < 0.12) {
    speed += 0.15;
  }
  updateScore();

  if (checkCollision()) {
    endGame();
    return;
  }

  animationId = requestAnimationFrame(update);
};

const startGame = () => {
  resetGame();
  overlay.classList.add("overlay--hidden");
  gameOverOverlay.classList.add("overlay--hidden");
  isRunning = true;
  animationId = requestAnimationFrame(update);
};

const endGame = () => {
  isRunning = false;
  if (animationId) {
    cancelAnimationFrame(animationId);
  }
  finalScoreEl.textContent = Math.floor(score).toString();
  gameOverOverlay.classList.remove("overlay--hidden");
};

const jump = () => {
  if (!isRunning) {
    return;
  }
  if (player.isGrounded) {
    player.velocityY = lift;
    player.isGrounded = false;
  }
};

startBtn.addEventListener("click", startGame);
restartBtn.addEventListener("click", startGame);

window.addEventListener("keydown", (event) => {
  if (event.code !== "Space") {
    return;
  }
  event.preventDefault();
  if (!isRunning) {
    startGame();
  } else {
    jump();
  }
  spacePressed = true;
});

window.addEventListener("keyup", (event) => {
  if (event.code === "Space") {
    spacePressed = false;
  }
});

resetGame();
drawScene();
