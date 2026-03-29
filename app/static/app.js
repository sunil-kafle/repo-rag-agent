const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");
const chatMessages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderSimpleMarkdown(text) {
  let html = escapeHtml(text);

  html = html.replace(
    /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
  );

  html = html.replace(/\n/g, "<br>");
  return html;
}

function createMessageElement(role, text, renderMarkdown = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role === "user" ? "user-message" : "assistant-message"}`;

  const icon = document.createElement("div");
  icon.className = `message-icon ${role === "user" ? "user-icon" : "assistant-icon"}`;

  const img = document.createElement("img");
  img.className = "message-icon-img";
  img.src = role === "user" ? "/static/human.png" : "/static/robot.png";
  img.alt = role === "user" ? "User icon" : "Assistant icon";
  icon.appendChild(img);

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";

  if (renderMarkdown) {
    bubble.innerHTML = renderSimpleMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  wrapper.appendChild(icon);
  wrapper.appendChild(bubble);

  return wrapper;
}

function appendMessage(role, text, renderMarkdown = false) {
  const element = createMessageElement(role, text, renderMarkdown);
  chatMessages.appendChild(element);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return element;
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage("user", question, false);
  questionInput.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "Sending...";

  const loadingMessage = appendMessage("assistant", "Thinking...", false);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        question: question,
        debug: false
      })
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    const answer = data.answer || "No answer returned.";

    loadingMessage.remove();
    appendMessage("assistant", answer, true);
  } catch (error) {
    loadingMessage.remove();
    appendMessage("assistant", `Error: ${error.message}`, false);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
    questionInput.focus();
  }
});