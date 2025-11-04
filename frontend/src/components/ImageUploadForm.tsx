import type { ChangeEvent, DragEvent } from 'react';
import { useEffect, useRef, useState } from 'react';

const MAX_FILE_SIZE_MB = 15;

const ImageUploadForm = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const captureInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return undefined;
    }

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const validateAndSetFile = (nextFile: File | undefined | null) => {
    if (!nextFile) {
      return;
    }

    if (!nextFile.type.startsWith('image/')) {
      setError('Поддерживаются только изображения.');
      return;
    }

    if (nextFile.size > MAX_FILE_SIZE_MB * 1024 * 1024) {
      setError(`Размер файла не должен превышать ${MAX_FILE_SIZE_MB} МБ.`);
      return;
    }

    setError(null);
    setFile(nextFile);
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const next = event.target.files?.[0];
    validateAndSetFile(next ?? null);
  };

  const handleDrop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
    const next = event.dataTransfer.files?.[0];
    validateAndSetFile(next ?? null);
  };

  const handleDragOver = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault();
    event.stopPropagation();
  };

  const clearSelection = () => {
    setFile(null);
    setError(null);
    if (uploadInputRef.current) {
      uploadInputRef.current.value = '';
    }
    if (captureInputRef.current) {
      captureInputRef.current.value = '';
    }
  };

  return (
    <form className="upload" onSubmit={(event) => event.preventDefault()}>
      <header className="upload__header">
        <h2 className="upload__title">Определение породы</h2>
        <p className="upload__description">
          Загрузите фото собаки или сделайте снимок с камеры. Мы подберём породу и дадим рекомендации по уходу.
        </p>
      </header>

      <label
        className={file ? 'upload__area upload__area--filled' : 'upload__area'}
        htmlFor="upload-input"
        onDrop={handleDrop}
        onDragOver={handleDragOver}
      >
        {previewUrl ? (
          <img className="upload__preview" src={previewUrl} alt="Выбранное изображение" />
        ) : (
          <div className="upload__placeholder">
            <span className="upload__icon" aria-hidden="true">
              [photo]
            </span>
            <p>Перетащите фото сюда или воспользуйтесь кнопками ниже.</p>
          </div>
        )}
      </label>

      <input
        id="upload-input"
        ref={uploadInputRef}
        type="file"
        accept="image/*"
        className="upload__input"
        onChange={handleFileChange}
      />
      <input
        id="capture-input"
        ref={captureInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="upload__input"
        onChange={handleFileChange}
      />

      <div className="upload__actions">
        <label className="button button--primary" htmlFor="upload-input">
          Загрузить фото
        </label>
        <label className="button button--ghost" htmlFor="capture-input">
          Сделать снимок
        </label>
        {file ? (
          <button className="button button--ghost" type="button" onClick={clearSelection}>
            Очистить
          </button>
        ) : null}
      </div>

      {error ? <p className="upload__error">{error}</p> : null}

      <div className="upload__cta">
        <button className="button button--primary" type="button" disabled={!file}>
          Получить породу
        </button>
        <button className="button button--ghost" type="button" disabled={!file}>
          Рекомендации по уходу
        </button>
      </div>
    </form>
  );
};

export default ImageUploadForm;
