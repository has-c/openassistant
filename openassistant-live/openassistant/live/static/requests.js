'use strict';

// Pi extension requests retain their native types. Never auto-select consent.
function renderPiRequest(ticket, request, respond, index) {
  const title = document.createElement('strong'); title.textContent = request.title; ticket.append(title);
  if (request.message) { const detail = document.createElement('p'); detail.textContent = request.message; ticket.append(detail); }
  const finish = value => {
    respond(value);
    ticket.querySelectorAll('button,input,textarea,select').forEach(control => control.disabled = true);
  };
  const form = document.createElement('form'); form.className = 'live-request-form';
  if (request.method === 'confirm') {
    for (const [label, value] of [['Decline', false], ['Confirm', true]]) {
      const button = document.createElement('button'); button.type = value ? 'submit' : 'button'; button.textContent = label;
      if (!value) button.onclick = () => finish(false);
      form.append(button);
    }
    form.onsubmit = event => { event.preventDefault(); finish(true); };
  } else {
    const field = document.createElement(request.method === 'select' ? 'select' : 'textarea');
    field.id = `request-value-${index}`;
    const label = document.createElement('label'); label.htmlFor = field.id; label.textContent = 'Your answer';
    if (request.method === 'select') {
      const empty = document.createElement('option'); empty.value = ''; empty.textContent = 'Choose an option'; empty.disabled = true; empty.selected = true; field.append(empty);
      for (const [i, value] of request.options.entries()) {
        const option = document.createElement('option'); option.value = String(i); option.textContent = value; field.append(option);
      }
      field.required = true;
    } else { field.value = request.prefill || ''; field.placeholder = request.placeholder || ''; field.maxLength = 8000; }
    const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Send answer';
    form.append(label, field, submit);
    form.onsubmit = event => { event.preventDefault(); finish(request.method === 'select' ? request.options[Number(field.value)] : field.value); };
  }
  const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = 'Cancel'; cancel.onclick = () => finish(null);
  form.append(cancel); ticket.append(form);
}
