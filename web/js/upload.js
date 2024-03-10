'use strict';

const Base64Image = ({ encodedImage, ...rest }) => {
    return (
        <div {...rest}>
            <div className="img-container">
                {encodedImage
                    ? <img src={`data:image/jpeg;base64,${encodedImage}`} />
                    : <span>Your image goes here</span>
                }
            </div>

        </div>
    )
}

const FileUploadForm = ({ uploadUrl, setResults }) => {
    const [file, setFile] = React.useState(null);
    const [loading, setLoading] = React.useState(false);
    const [temp, setTemp] = React.useState(null);

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        setFile(selectedFile);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        if (!file) {
            console.error('Please select a file');
            return;
        }

        setLoading(true);

        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = async () => {
            const base64data = reader.result.split(',')[1];
            const data = { file: base64data };

            setTemp(JSON.stringify(base64data))

            try {
                const response = await fetch(uploadUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        limit: 3,
                        data: base64data,
                    }),
                });

                if (response.ok) {
                    setResults(await response.json())
                    console.error('File uploaded successfully');
                    return
                }

                console.error('Failed to upload file');
            } catch (error) {
                console.error('Error uploading file:', error);
                console.error('Error uploading file');
            } finally {
                setLoading(false);
            }
        };
    };

    return (
        <div className="image-upload__container">
            <form
                className="image-upload__form"
                onSubmit={handleSubmit}
            >
                <div className="form-input">
                    <input className="hidden" type="file" onChange={handleFileChange} id="actual-btn" />
                    <label htmlFor="actual-btn">Choose File</label>
                    <span id="file-chosen">{!!file ? file.name : "No file chosen"}</span>
                </div>

                <div className="form-input">
                    <button type="submit" disabled={loading}>
                        {loading ? 'Uploading...' : 'Upload'}
                    </button>
                </div>
            </form >

            <Base64Image
                className="image-upload__preview"
                encodedImage={JSON.parse(temp)}
            />

        </div >
    );
}
