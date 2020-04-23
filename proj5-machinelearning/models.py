import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0.0 :
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        success = False
        while success == False:
            success = True
            for x,y in dataset.iterate_once(1):
                if nn.as_scalar(y) == self.get_prediction(x):
                    continue
                else:
                    success = False
                    self.w.update(x, nn.as_scalar(y))

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dimension = 30
        self.w0 = nn.Parameter(1, self.dimension)
        self.b0 = nn.Parameter(1, self.dimension)
        self.w1 = nn.Parameter(self.dimension, 1)
        self.b1 = nn.Parameter(1, 1)
        self.tol = 0.02
        self.delta = -0.007

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        l = nn.Linear(x, self.w0)
        r = nn.ReLU(nn.AddBias(l, self.b0))
        l2 = nn.Linear(r, self.w1)
        ret = nn.AddBias(l2, self.b1)
        return ret

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(x)
        return nn.SquareLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])

                self.w0.update(grad[0], self.delta)
                self.w1.update(grad[1], self.delta)
                self.b0.update(grad[2], self.delta)
                self.b1.update(grad[3], self.delta)
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) >= self.tol:
                continue
            return

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.dimension = 100
        self.w0 = nn.Parameter(784, self.dimension)
        self.b0 = nn.Parameter(1, self.dimension)
        self.w1 = nn.Parameter(self.dimension, 10)
        self.b1 = nn.Parameter(1, 10)
        self.delta = -0.009

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        xw1 = nn.Linear(x, self.w0)
        ad1 = nn.AddBias(xw1, self.b0)
        r1 = nn.ReLU(ad1)
        xw2 = nn.Linear(r1, self.w1)
        ad2 = nn.AddBias(xw2, self.b1)
        return ad2



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(x)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(60):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss, [self.w0, self.w1, self.b0, self.b1])

                self.w0.update(grad[0], self.delta)
                self.w1.update(grad[1], self.delta)
                self.b0.update(grad[2], self.delta)
                self.b1.update(grad[3], self.delta)

            if dataset.get_validation_accuracy() < 0.975:
                continue
            return

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        self.batch_size = 10
        self.hiddendim = 400
        self.delta = -0.004
        self.wh1 = nn.Parameter(self.hiddendim, self.hiddendim)
        self.wh2 = nn.Parameter(self.hiddendim, self.hiddendim)   
        self.w = nn.Parameter(self.num_chars, self.hiddendim)        
        self.wf = nn.Parameter(self.hiddendim, 5)            

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.Linear(xs[0], self.w)
        for x in xs[1:]:
            z = nn.Add(nn.Linear(x, self.w), nn.Linear(z, self.wh1))
        r = nn.ReLU(z)
        z = nn.Linear(r,self.wh2)
        r = nn.ReLU(z)

        return nn.Linear(r, self.wf)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        pred = self.run(xs)
        return nn.SoftmaxLoss(pred, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                grad = nn.gradients(loss, [self.w, self.wh1,self.wh2, self.wf])
                self.w.update(grad[0], self.delta)
                self.wh1.update(grad[1], self.delta)
                self.wh2.update(grad[2], self.delta)
                self.wf.update(grad[3], self.delta)
            if dataset.get_validation_accuracy() < 0.85:
                continue
            return
