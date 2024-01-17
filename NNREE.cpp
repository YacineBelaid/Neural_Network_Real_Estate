/*
    Pour compiler:
    g++ tp4.cpp -larmadillo -o tp4
    ou
    # Si mlpack n'a pas Ã©tÃ© "installÃ©" avec `make install`
    g++ tp4.cpp -I ../mlpack-4.2.0/src/ -larmadillo -o tp4
*/

#include <mlpack.hpp>
using namespace mlpack;

double ComputeMSE(const arma::mat& pred, const arma::mat& Y){
    return  SquaredEuclideanDistance::Evaluate(pred , Y) / pred.n_cols;
}

int main(){
    // 1. Charger les donnÃ©es
    arma::mat mentrainement;
    data::DatasetInfo info;
    const char* nomFichierEntrainement = "maisons-entrainement.csv";
    if(!data::Load(nomFichierEntrainement, mentrainement, info)){
        std::cout << "Erreur chargement " << nomFichierEntrainement << std::endl;
        return 1;
    }
    arma::mat mevaluation;
    const char* nomFichierEvaluation = "maisons-evaluation.csv";
    if(!data::Load(nomFichierEvaluation, mevaluation, info)){
        std::cout << "Erreur chargement " << nomFichierEvaluation << std::endl;
        return 1;
    }

    // 2. CrÃ©er les X (vecteurs d'entrÃ©es) et Y (sorties) sous forme de matrices.
    // Attention: dans mlpack, les matrices sont transposÃ©es.
    arma::mat entrainementX = mentrainement.submat(0, 0, mentrainement.n_rows-2, mentrainement.n_cols-1);
    arma::mat evaluationX = mevaluation.submat(0, 0, mevaluation.n_rows-2, mevaluation.n_cols-1);

    arma::mat entrainementY = mentrainement.row(mentrainement.n_rows-1);
    arma::mat evaluationY = mevaluation.row(mentrainement.n_rows-1);

    // 3. Normaliser les donnÃ©es dans des intervales [0, 1].
    data::MinMaxScaler scaleX;
    // Scaler for predictions.
    data::MinMaxScaler scaleY;
    // Fit scaler only on training data.
    scaleX.Fit(entrainementX);
    scaleX.Transform(entrainementX, entrainementX);
    scaleX.Transform(evaluationX, evaluationX);

    // Scale training predictions.
    scaleY.Fit(entrainementY);
    scaleY.Transform(entrainementY, entrainementY);
    scaleY.Transform(evaluationY, evaluationY);


    // 4. Architecture du RÃ©seau de neurones artificiels.
    FFN<MeanSquaredError, XavierInitialization> model;
        // Couche entree
		model.Add<Linear>(128);
		model.Add<LeakyReLU>();

		model.Add<Linear>(64);
		model.Add<LeakyReLU>();

		model.Add<Linear>(32);
		model.Add<LeakyReLU>();

		// Chouche sorti
		model.Add<Linear>(1);
    // 5. Entrainement: choix d'hyper-paramÃ¨tres
  const int EPOCHS = 1000;
  constexpr double STEP_SIZE = 0.0002;
  constexpr int BATCH_SIZE = 32;
  constexpr double STOP_TOLERANCE = 0.0000000001;

    ens::Adam optimisation(
        STEP_SIZE,
        BATCH_SIZE,
        0.9,
        0.999,
        1e-8,
        EPOCHS*BATCH_SIZE,
        STOP_TOLERANCE,
        true);


    model.Train(entrainementX, entrainementY, optimisation);

    // 6. Ã‰valuation
    arma::mat predictions;
    model.Predict(evaluationX, predictions);
    scaleY.InverseTransform(predictions, predictions);

    // 7. Afficher les prix prÃ©dit vs prix "rÃ©els" (% dÃ©viation)
  std::cout << "ID\tPrédiction\tVérité\tÉcart(%)" << std::endl;
for (int i = 0; i < predictions.n_cols; i++) {
    double verite = mevaluation(mevaluation.n_rows - 1, i);
    double prediction = predictions(0, i);
    double ecart = ((prediction - verite) / verite) * 100.0;

    std::cout << mevaluation(0, i) << '\t'
              << prediction << '\t'
              << verite << '\t'
              << ecart << '%' << std::endl;
}

    // 8. Afficher MSE
    double validMSE = ComputeMSE(predictions, evaluationY);
  std::cout << "Mean Squared Error on Prediction data points: " << validMSE << std::endl;
    //
    return 0;
}

