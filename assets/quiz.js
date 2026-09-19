// Reusable retrieval-practice quiz widget.
// Markup:
// <div class="quiz">
//   <p class="q">Question text?</p>
//   <div class="options">
//     <button class="opt" data-correct="true">Answer text</button>
//     <button class="opt">Distractor text</button>
//   </div>
//   <p class="feedback"></p>
// </div>
(function () {
  function initQuiz(quiz) {
    var buttons = quiz.querySelectorAll("button.opt");
    var feedback = quiz.querySelector(".feedback");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        if (quiz.dataset.answered) return;
        quiz.dataset.answered = "true";
        var isCorrect = btn.dataset.correct === "true";
        buttons.forEach(function (b) {
          b.disabled = true;
          if (b.dataset.correct === "true") b.classList.add("correct");
        });
        if (!isCorrect) btn.classList.add("incorrect");
        if (feedback) {
          feedback.textContent = isCorrect
            ? "Correct."
            : "Not quite — the highlighted option is correct.";
          feedback.classList.add(isCorrect ? "correct" : "incorrect");
        }
      });
    });
  }
  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".quiz").forEach(initQuiz);
  });
})();
