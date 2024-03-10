'use strict';

const UPLOAD_URL = "http://localhost:8080/classify"

const TEST_DATA = [
    {
        "model": "resnet50-caffe2-v1-9.onnx",
        "probs": [
            {
                "class": "n03691459 loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
                "prob": 0.7141492962837219,
            },
            {
                "class": "n04286575 spotlight, spot",
                "prob": 0.11104119569063187
            },
            {
                "class": "n02999410 chain",
                "prob": 0.10356773436069489
            }
        ],
        "top_full_label": "n02129604 tiger, Panthera tigris",
        "time_spent": {
            "span": 109517,
            "unit": "microsecond"
        }
    }
]

const App = () => {
    const [results, setResults] = React.useState(TEST_DATA)

    return (
        <div>
            <h3>File upload</h3>
            <FileUploadForm
                setResults={setResults}
                uploadUrl={UPLOAD_URL}
            />

            <h3>Results</h3>
            <Charts results={results} />
        </div>
    )
}

const domContainer = document.querySelector('#root');
const root = ReactDOM.createRoot(domContainer);
root.render(React.createElement(App));