'use strict';

const TEST_DATA = [
    {
        "model": "resnet50-caffe2-v1-9.onnx",
        "probs": [
            {
                "class": "n03691459 loudspeaker, speaker, speaker unit, loudspeaker system, speaker system",
                "prob": 0.7141492962837219
            },
            {
                "class": "n04286575 spotlight, spot",
                "prob": 0.11104119569063187
            },
            {
                "class": "n02999410 chain",
                "prob": 0.10356773436069489
            },
            {
                "class": "n04485082 tripod",
                "prob": 0.06055587902665138
            },
            {
                "class": "n04153751 screw",
                "prob": 0.007883824408054352
            },
            {
                "class": "n09229709 bubble",
                "prob": 0.0015070264926180243
            },
            {
                "class": "n03041632 cleaver, meat cleaver, chopper",
                "prob": 0.0008784943493083119
            },
            {
                "class": "n03814906 necklace",
                "prob": 0.00027209086692892015
            }
        ],
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
                uploadUrl={"http://localhost:8080/classify"}
            />

            <h3>Results</h3>
            <Charts results={results} />
        </div>
    )
}

const domContainer = document.querySelector('#root');
const root = ReactDOM.createRoot(domContainer);
root.render(React.createElement(App));