const submitButton = document.getElementById('submit');
const promptInput = document.getElementById('prompt');
const answerOutput = document.getElementById('answer');
const contextOutput = document.getElementById('context');
const debugCheckbox = document.getElementById('debug');

const apiUrl = '/api/query';

const statusIndicator = document.getElementById('status');

submitButton.addEventListener('click', async () => {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    answerOutput.textContent = 'Please enter a question first.';
    statusIndicator.textContent = 'Ready';
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = 'Thinking...';
  statusIndicator.textContent = 'Processing request...';
  answerOutput.textContent = 'Thinking...';
  contextOutput.textContent = 'Retrieving context...';

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, debug: debugCheckbox.checked }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Query request failed');
    }

    const data = await response.json();
    answerOutput.textContent = data.answer;
    contextOutput.textContent = data.context;
    statusIndicator.textContent = 'Completed';
  } catch (err) {
    answerOutput.textContent = `Error: ${err.message}`;
    contextOutput.textContent = '';
    statusIndicator.textContent = 'Failed';
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = 'Ask';
  }
});
