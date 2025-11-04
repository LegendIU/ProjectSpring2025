import PageSection from '../components/PageSection';

const capabilities = [
  'Modeling: EfficientNet, ConvNeXt, custom hybrid transformers',
  'Acceleration: ONNX Runtime, TensorRT, quantization aware training',
  'Pipelines: clear data lineage, automated retraining, feature stores',
  'Delivery: FastAPI, gRPC, real-time dashboards, mobile SDK handoff'
];

const AboutPage = () => {
  return (
    <div className="page-stack">
      <PageSection
        subtitle="About"
        title="K3Project by Nik52"
        description="The lab is a playground for applied computer vision and ML infra. I merge research depth with product pragmatism."
      >
        <div className="card-grid">
          {capabilities.map((item) => (
            <article className="card" key={item}>
              <p className="card__description">{item}</p>
            </article>
          ))}
        </div>
      </PageSection>

      <PageSection
        subtitle="Background"
        title="From research to reliable delivery"
        description="Over the past five years I have led initiatives in detection, generative augmentation and low-latency inference."
      >
        <div className="timeline">
          <div className="timeline__item">
            <h3 className="timeline__title">Academic roots</h3>
            <p className="timeline__meta">Deep learning · Vision systems</p>
            <p className="card__description">
              I started the K3Project as a research portfolio, testing novel augmentation stacks across medical and
              industrial datasets.
            </p>
          </div>
          <div className="timeline__item">
            <h3 className="timeline__title">Product leadership</h3>
            <p className="timeline__meta">Cross-functional teams</p>
            <p className="card__description">
              Transitioned prototypes into production services, aligning data scientists, engineers and delivery
              managers on measurable goals.
            </p>
          </div>
          <div className="timeline__item">
            <h3 className="timeline__title">Future roadmap</h3>
            <p className="timeline__meta">Edge CV · User tooling</p>
            <p className="card__description">
              Building a toolkit around K3Project for rapid dataset iteration, low-code experimentation and controlled
              rollouts.
            </p>
          </div>
        </div>
      </PageSection>
    </div>
  );
};

export default AboutPage;
