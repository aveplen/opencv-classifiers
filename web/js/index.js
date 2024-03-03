'use strict';

const App = () => {
    const [counter, setCounter] = React.useState(0)
    const increment = () => { setCounter(counter + 1) }
    const decrement = () => { setCounter(counter - 1) }
    return (
        <div>
            <p class="notice">
                this is a notice box
            </p>

            <article>
                <h2>This is an article</h2>
                <p>Some content will go here, which will be inside your article.</p>
            </article>

            <div className="container">
                <span>{ counter }</span>
            </div>

             <br />

            <div className="container">
                <button onClick={ decrement }> - </button>
                <button onClick={ increment }> + </button>
            </div>
            
        </div>
    )
}

const domContainer = document.querySelector('#root');
const root = ReactDOM.createRoot(domContainer);
root.render(React.createElement(App));