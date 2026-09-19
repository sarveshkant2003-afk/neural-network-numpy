# Neural Networks from Scratch (NumPy) Resources

## Knowledge

- [Book: _Neural Networks and Deep Learning_ by Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
  Free online book that builds a from-scratch (numpy-based) MNIST digit classifier chapter by chapter, deriving backpropagation from first principles alongside the code. Use for: the core mission path — architecture, backprop derivation, training loop design. This is the primary spine of the course.
- [Video series: "Neural Networks" by 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  Four videos: what a neural network is, gradient descent, what backprop is really doing, and the backprop calculus. Best visual intuition for *why* backprop's chain rule looks the way it does. Use for: building intuition before/alongside the math-heavy derivation.
- [CS231n notes: "Backpropagation, Intuitions"](https://cs231n.github.io/optimization-2/)
  Stanford's course notes on backprop as recursive chain-rule application over a computational graph. More rigorous/compact than Nielsen's treatment. Use for: double-checking gradient derivations, especially once matrix-shaped gradients get confusing.
- [NumPy quickstart (official docs)](https://numpy.org/doc/stable/user/quickstart.html)
  The canonical tutorial for array creation, shapes, indexing, and broadcasting. Use for: the numpy-fundamentals lesson, since this is a stated gap.
- [NumPy: the absolute basics for beginners (official docs)](https://numpy.org/doc/stable/user/absolute_beginners.html)
  Even more gradual on-ramp than the quickstart. Use for: first exposure if the quickstart moves too fast.

## Wisdom (Communities)

- [r/MachineLearning](https://reddit.com/r/MachineLearning)
  Large, research-leaning community. Use for: once the MNIST classifier works, posting/discussing the implementation and getting critique on things like initialization, vectorization style, or numerical stability.
- [r/learnmachinelearning](https://reddit.com/r/learnmachinelearning)
  Beginner-friendlier than r/MachineLearning, high traffic. Use for: troubleshooting stuck points, "does this gradient look right" sanity checks, and seeing how others approached the same from-scratch build.

## Gaps

- No community/resource yet identified specifically for numpy performance/vectorization code review (as opposed to ML theory). Revisit once the implementation is running and the question becomes "is this numpy idiomatic," not "is this math correct."
